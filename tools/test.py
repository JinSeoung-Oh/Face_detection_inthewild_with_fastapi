# Copyright 2022 The vedadet Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.


import argparse

import torch

from vedacore.fileio import dump
from vedacore.misc import Config, ProgressBar, load_weights
from vedacore.parallel import MMDataParallel
from vedadet.datasets import build_dataloader, build_dataset
from vedadet.engines import build_engine


def parse_args():
    parser = argparse.ArgumentParser(description='Test a detector')
    parser.add_argument('config', help='test config file path')
    parser.add_argument('checkpoint', help='checkpoint file')
    parser.add_argument('--out', help='output result file in pickle format')

    args = parser.parse_args()
    return args


def prepare(cfg, checkpoint):

    engine = build_engine(cfg.val_engine)
    load_weights(engine.model, checkpoint, map_location='cpu')

    device = torch.cuda.current_device()
    engine = MMDataParallel(
        engine.to(device), device_ids=[torch.cuda.current_device()])

    dataset = build_dataset(cfg.data.val, dict(test_mode=True))
    dataloader = build_dataloader(dataset, 1, 1, dist=False, shuffle=False)

    return engine, dataloader


def test(engine, data_loader):
    engine.eval()
    results = []
    dataset = data_loader.dataset
    prog_bar = ProgressBar(len(dataset))
    for i, data in enumerate(data_loader):

        with torch.no_grad():
            result = engine(data)[0]

        results.append(result)
        batch_size = len(data['img_metas'][0].data)
        for _ in range(batch_size):
            prog_bar.update()
    return results


def main():

    args = parse_args()
    cfg = Config.fromfile(args.config)

    if args.out is not None and not args.out.endswith(('.pkl', '.pickle')):
        raise ValueError('The output file must be a pkl file.')

    engine, data_loader = prepare(cfg, args.checkpoint)

    results = test(engine, data_loader)

    if args.out:
        print(f'\nwriting results to {args.out}')
        dump(results, args.out)

    data_loader.dataset.evaluate(results)


if __name__ == '__main__':
    main()
