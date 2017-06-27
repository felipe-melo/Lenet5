from ReadDatabase import download_no_mnist, maybe_extract


def main():
    train_filename = download_no_mnist('notMNIST_large.tar.gz', 247336696)
    test_filename = download_no_mnist('notMNIST_small.tar.gz', 8458043)

    train_folders = maybe_extract(train_filename)
    test_folders = maybe_extract(test_filename)


if "__main__" == __name__:
    main()
