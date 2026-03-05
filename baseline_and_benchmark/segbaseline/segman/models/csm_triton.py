# models/csm_triton.py
import torch
import triton
import triton.language as tl


@triton.jit
def cross_scan_kernel(
        x_ptr,  # 输入张量指针
        y_ptr,  # 输出张量指针
        B,  # 批次大小
        C,  # 通道数
        H,  # 高度
        W,  # 宽度
        stride_xb, stride_xc, stride_xh, stride_xw,
        stride_yb, stride_yc, stride_yh, stride_yw,
        BLOCK_SIZE: tl.constexpr,
):
    # 计算像素索引
    pid = tl.program_id(0)
    num_pixels = H * W

    # 计算批次和通道索引
    b = pid // (4 * num_pixels)
    direction = (pid % (4 * num_pixels)) // num_pixels
    pixel_idx = pid % num_pixels

    # 计算像素坐标
    h = pixel_idx // W
    w = pixel_idx % W

    # 根据扫描方向调整坐标
    if direction == 0:  # 左上到右下
        pass  # 保持原始坐标
    elif direction == 1:  # 右上到左下
        w = W - 1 - w
    elif direction == 2:  # 左下到右上
        h = H - 1 - h
    elif direction == 3:  # 右下到左上
        h = H - 1 - h
        w = W - 1 - w

    # 计算输入指针偏移
    x_offset = b * stride_xb + h * stride_xh + w * stride_xw

    # 计算输出指针偏移
    y_offset = b * stride_yb + direction * stride_yc + h * stride_yh + w * stride_yw

    # 加载和存储数据
    for c in range(0, C, BLOCK_SIZE):
        # 计算通道偏移
        c_offs = c + tl.arange(0, BLOCK_SIZE)
        mask = c_offs < C

        # 加载输入数据
        x_val = tl.load(x_ptr + x_offset + c_offs * stride_xc, mask=mask)

        # 存储到输出
        tl.store(y_ptr + y_offset + c_offs * stride_yc, x_val, mask=mask)


@triton.jit
def cross_merge_kernel(
        x_ptr,  # 输入张量指针
        y_ptr,  # 输出张量指针
        B,  # 批次大小
        C,  # 通道数
        H,  # 高度
        W,  # 宽度
        stride_xb, stride_xc, stride_xh, stride_xw,
        stride_yb, stride_yc, stride_yh, stride_yw,
        BLOCK_SIZE: tl.constexpr,
):
    # 计算像素索引
    pid = tl.program_id(0)
    num_pixels = H * W

    # 计算批次和通道索引
    b = pid // num_pixels
    pixel_idx = pid % num_pixels

    # 计算像素坐标
    h = pixel_idx // W
    w = pixel_idx % W

    # 计算输出指针偏移
    y_offset = b * stride_yb + h * stride_yh + w * stride_yw

    # 初始化累加器
    acc = tl.zeros((BLOCK_SIZE,), dtype=tl.float32)

    # 遍历四个方向
    for direction in range(4):
        # 根据扫描方向调整坐标
        if direction == 0:  # 左上到右下
            h_dir = h
            w_dir = w
        elif direction == 1:  # 右上到左下
            h_dir = h
            w_dir = W - 1 - w
        elif direction == 2:  # 左下到右上
            h_dir = H - 1 - h
            w_dir = w
        elif direction == 3:  # 右下到左上
            h_dir = H - 1 - h
            w_dir = W - 1 - w

        # 计算输入指针偏移
        x_offset = b * stride_xb + direction * stride_xc + h_dir * stride_xh + w_dir * stride_xw

        # 加载数据并累加
        for c in range(0, C, BLOCK_SIZE):
            # 计算通道偏移
            c_offs = c + tl.arange(0, BLOCK_SIZE)
            mask = c_offs < C

            # 加载输入数据
            x_val = tl.load(x_ptr + x_offset + c_offs * stride_xc, mask=mask)

            # 累加到结果
            acc += tl.where(mask, x_val, 0.0)

    # 平均并存储结果
    acc = acc / 4.0

    # 存储到输出
    for c in range(0, C, BLOCK_SIZE):
        # 计算通道偏移
        c_offs = c + tl.arange(0, BLOCK_SIZE)
        mask = c_offs < C

        # 存储结果
        tl.store(y_ptr + y_offset + c_offs * stride_yc, acc, mask=mask)


class CrossScanTriton(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x):
        # 输入形状: (B, C, H, W)
        B, C, H, W = x.shape

        # 输出形状: (B, 4, C, H, W)
        y = torch.empty(B, 4, C, H, W, dtype=x.dtype, device=x.device)

        # 计算步长
        stride_xb, stride_xc, stride_xh, stride_xw = x.stride()
        stride_yb, stride_yc, stride_yh, stride_yw = y.stride()

        # 网格大小
        grid = lambda meta: (B * 4 * H * W,)

        # 调用Triton内核
        cross_scan_kernel[grid](
            x, y,
            B, C, H, W,
            stride_xb, stride_xc, stride_xh, stride_xw,
            stride_yb, stride_yc, stride_yh, stride_yw,
            BLOCK_SIZE=128
        )

        ctx.save_for_backward(x)
        ctx.B, ctx.C, ctx.H, ctx.W = B, C, H, W

        return y

    @staticmethod
    def backward(ctx, grad_y):
        # 使用CrossMergeTriton进行反向传播
        x, = ctx.saved_tensors
        B, C, H, W = ctx.B, ctx.C, ctx.H, ctx.W

        # 计算梯度
        grad_x = CrossMergeTriton.apply(grad_y)

        return grad_x


class CrossMergeTriton(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x):
        # 输入形状: (B, 4, C, H, W)
        B, D, C, H, W = x.shape
        assert D == 4, "输入张量的第二维度必须为4"

        # 输出形状: (B, C, H, W)
        y = torch.empty(B, C, H, W, dtype=x.dtype, device=x.device)

        # 计算步长
        stride_xb, stride_xc, stride_xh, stride_xw = x.stride()
        stride_yb, stride_yc, stride_yh, stride_yw = y.stride()

        # 网格大小
        grid = lambda meta: (B * H * W,)

        # 调用Triton内核
        cross_merge_kernel[grid](
            x, y,
            B, C, H, W,
            stride_xb, stride_xc, stride_xh, stride_xw,
            stride_yb, stride_yc, stride_yh, stride_yw,
            BLOCK_SIZE=128
        )

        ctx.save_for_backward(x)
        ctx.B, ctx.C, ctx.H, ctx.W = B, C, H, W

        return y

    @staticmethod
    def backward(ctx, grad_y):
        # 使用CrossScanTriton进行反向传播
        x, = ctx.saved_tensors
        B, C, H, W = ctx.B, ctx.C, ctx.H, ctx.W

        # 计算梯度
        grad_x = CrossScanTriton.apply(grad_y)

        return grad_x


# 封装为PyTorch模块
class CrossScanTritonModule(nn.Module):
    def forward(self, x):
        return CrossScanTriton.apply(x)


class CrossMergeTritonModule(nn.Module):
    def forward(self, x):
        return CrossMergeTriton.apply(x)