from mnist.dataset import MnistDataset

def test_data():
    train = MnistDataset(train=True)
    test = MnistDataset(train=False)
    assert len(train) == 30000
    assert len(test) == 5000
    for dataset in [train, test]:
        for x, y in dataset:
            assert x.shape == (1, 28, 28)
            assert y in range(10)
    # train_targets = torch.unique(train.target[1])
    # assert (train_targets == torch.arange(0,10)).all()
    # test_targets = torch.unique(test.target[1])
    # assert (test_targets == torch.arange(0,10)).all()
