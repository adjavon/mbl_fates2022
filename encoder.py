import torch


class Encoder(torch.nn.Module):

    def __init__(
            self,
            input_size,
            fmaps=12,
            downsample_factors=[(2, 2), (2, 2), (3, 3), (3, 3)],
            output_classes=32):

        super().__init__()

        self.input_size = input_size

        current_fmaps, h, w = tuple(input_size)
        current_size = (h, w)

        features = []
        for i in range(len(downsample_factors)):

            features += [
                torch.nn.Conv2d(
                    current_fmaps,
                    fmaps,
                    kernel_size=3,
                    padding=1),
                torch.nn.BatchNorm2d(fmaps),
                torch.nn.ReLU(),
                torch.nn.Conv2d(
                    fmaps,
                    fmaps,
                    kernel_size=3,
                    padding=1),
                torch.nn.BatchNorm2d(fmaps),
                torch.nn.ReLU(),
                torch.nn.MaxPool2d(downsample_factors[i])
            ]

            current_fmaps = fmaps
            fmaps *= 2

            size = tuple(
                int(c/d)
                for c, d in zip(current_size, downsample_factors[i]))
            check = (
                s*d == c
                for s, d, c in zip(size, downsample_factors[i], current_size))
            assert all(check), \
                "Can not downsample %s by chosen downsample factor" % \
                (current_size,)
            current_size = size

        self.features = torch.nn.Sequential(*features)

        classifier = [
            torch.nn.Linear(
                current_size[0] *
                current_size[1] *
                current_fmaps,
                output_classes)]

        self.classifier = torch.nn.Sequential(*classifier)

    def forward(self, raw):
        # add a channel dimension to raw
        # shape = tuple(raw.shape)
        # raw = raw.reshape(shape[0], 1, shape[1], shape[2])

        # compute features
        f = self.features(raw)
        f = f.view(f.size(0), -1)

        # classify
        y = self.classifier(f)
        return y
