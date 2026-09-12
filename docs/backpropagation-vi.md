# Lan truyền ngược trong CNN — giải thích từng lớp

Tài liệu này đi cùng package `cnn_backprop` và tập trung vào câu hỏi quan trọng nhất: **gradient đi từ loss về từng tham số như thế nào?** Mọi công thức dưới đây đều có implementation NumPy và được kiểm tra bằng finite differences.

## 1. Computational graph

Một CNN có thể xem như chuỗi hàm hợp:

$$
X \rightarrow \operatorname{Conv} \rightarrow \operatorname{ReLU}
\rightarrow \operatorname{Pool} \rightarrow \operatorname{Dense}
\rightarrow Z \rightarrow L
$$

Forward pass tính activation từ trái sang phải. Backward pass áp dụng chain rule từ phải sang trái:

$$
\frac{\partial L}{\partial X}
=
\frac{\partial L}{\partial Z}
\frac{\partial Z}{\partial X}
$$

Mỗi layer chỉ cần biết hai việc:

1. lưu những giá trị cần thiết trong forward pass;
2. nhận gradient từ layer phía sau và trả gradient cho layer phía trước.

## 2. Softmax và cross-entropy

Với logits $z \in \mathbb{R}^{C}$:

$$
p_i = \frac{e^{z_i}}{\sum_j e^{z_j}}
$$

Để tránh overflow, implementation trừ `max(logits)` trước khi lấy số mũ. Khi ghép softmax với cross-entropy, đạo hàm rút gọn thành:

$$
\frac{\partial L}{\partial z_i} = \frac{p_i-y_i}{N}
$$

$N$ là batch size. Phép chia này phải xuất hiện đúng một lần; thiếu nó làm gradient tăng theo batch size, còn chia lặp lại khiến mô hình học quá chậm.

## 3. Dense layer

Forward:

$$
Z=XW+b
$$

Backward, với $G=\partial L/\partial Z$:

$$
dW=X^TG, \qquad db=\sum_n G_n, \qquad dX=GW^T
$$

Layer chỉ tính và lưu `dweight`, `dbias`. Optimizer chịu trách nhiệm cập nhật tham số sau đó. Việc tách hai bước giúp gradient có thể được kiểm tra trước khi weight thay đổi.

## 4. ReLU

$$
\operatorname{ReLU}(x)=\max(0,x)
$$

Gradient chỉ đi qua vị trí có activation dương:

$$
\frac{\partial L}{\partial x}
=
\frac{\partial L}{\partial y}\cdot \mathbb{1}(x>0)
$$

Vì vậy forward pass lưu một boolean mask `x > 0`.

## 5. Max pooling

Max pooling chọn phần tử lớn nhất trong mỗi cửa sổ. Khi backward, toàn bộ gradient được chuyển về **đúng một vị trí argmax**; các phần tử còn lại nhận 0.

Nếu một cửa sổ $2\times2$ là

$$
\begin{bmatrix}1&3\\2&0\end{bmatrix}
$$

và gradient đầu ra bằng 1, gradient đầu vào là

$$
\begin{bmatrix}0&1\\0&0\end{bmatrix}.
$$

Không nên dùng mask `(window == max)` mà không xử lý tie, vì nhiều phần tử bằng nhau có thể nhân bản gradient.

## 6. Convolution / cross-correlation

Các framework deep learning thường triển khai phép *cross-correlation* nhưng vẫn gọi layer là convolution. Với output channel $o$:

$$
Y_{n,o,h,w}
=
\sum_{c,i,j}X_{n,c,h+i,w+j}W_{o,c,i,j}+b_o
$$

Từ upstream gradient $G=\partial L/\partial Y$:

$$
\frac{\partial L}{\partial W_{o,c,i,j}}
=
\sum_{n,h,w}G_{n,o,h,w}X_{n,c,h+i,w+j}
$$

$$
\frac{\partial L}{\partial b_o}=\sum_{n,h,w}G_{n,o,h,w}
$$

Gradient theo input được cộng dồn vì một pixel có thể thuộc nhiều receptive fields:

$$
\frac{\partial L}{\partial X_{n,c,h+i,w+j}}
\mathrel{+}=G_{n,o,h,w}W_{o,c,i,j}
$$

Trong `layers.py`, forward dùng `sliding_window_view` và `einsum` để biểu diễn phép tính rõ ràng; backward cộng các contribution theo từng vị trí kernel.

## 7. Gradient checking

Đạo hàm số bằng central difference:

$$
g_i^{num}=\frac{f(x_i+\epsilon)-f(x_i-\epsilon)}{2\epsilon}
$$

được so với gradient analytical bằng relative error:

$$
\operatorname{error}=
\max_i\frac{|g_i-g_i^{num}|}{\max(10^{-12},|g_i|+|g_i^{num}|)}.
$$

Với `epsilon=1e-5`, Conv2D trong project cho sai số khoảng $10^{-10}$—thấp hơn nhiều so với ngưỡng test $10^{-7}$.

Chạy kiểm tra:

```bash
python -m pip install -e ".[dev]"
python examples/gradient_check.py
pytest
```

## 8. Một training step hoàn chỉnh

```python
logits = model.forward(images)
loss, gradient = softmax_cross_entropy(logits, labels)
model.backward(gradient)
optimizer.step(model)
```

Ba dòng cuối tương ứng với ba khái niệm tách biệt:

1. **Loss derivative:** tạo gradient đầu tiên tại logits.
2. **Backpropagation:** tính gradient của mọi activation và tham số.
3. **Optimization:** dùng gradient để cập nhật tham số.

Sự tách biệt này là nền tảng của các framework hiện đại, dù phần autograd thường che giấu bước số 2.
