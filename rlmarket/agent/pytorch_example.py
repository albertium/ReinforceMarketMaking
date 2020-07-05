import torch
import numpy as np
import plotly.graph_objects as go
import time


def generate_data(n, lags):
    t = np.linspace(0, 2 * np.pi * n / 100, n)
    prices = np.sin(t) + np.random.randn(len(t)) * 0
    x = [prices[idx: idx + lags] for idx in range(n - lags)]
    y = prices[lags:][:, None]
    return t[lags:], x, y


if __name__ == '__main__':
    lags = 5
    learning_rate = 1e-4
    n, n_in, h, n_out = 500, lags, lags * 2, 1
    device = torch.device('cpu')

    # Model
    model = torch.nn.Sequential(
        torch.nn.Linear(n_in, h),
        torch.nn.LeakyReLU(0.01),
        torch.nn.Linear(h, n_out)
    ).to(device)

    opt = torch.optim.Adam(model.parameters(), lr=learning_rate)
    print(f'Number of parameters: {sum(x.numel() for x in model.parameters())}')

    start_time = time.time()
    train_set = generate_data(n, lags)

    for i in range(10000):
        # Data
        x = torch.from_numpy(np.array(train_set[1])).float().to(device)
        y = torch.from_numpy(np.array(train_set[2])).float().to(device)

        y_hat = model(x)
        loss = torch.nn.MSELoss()(y_hat, y)

        if i % 1000 == 0:
            print(i, loss.item())

        opt.zero_grad()
        loss.backward()
        opt.step()

    print(f'Training takes {time.time() - start_time}s')

    # Test
    test_set = generate_data(200, lags)
    x = torch.from_numpy(np.array(test_set[1])).float().to(device)
    y = torch.from_numpy(np.array(test_set[2])).float().to(device)
    y_pred = model(x)

    fig = go.Figure(data=[
        go.Scatter(x=test_set[0], y=test_set[2].flatten(), mode='lines'),
        go.Scatter(x=test_set[0], y=y_pred.detach().numpy().flatten(), mode='lines'),
    ])
    fig.write_html('abc.html')
    # fig.show()