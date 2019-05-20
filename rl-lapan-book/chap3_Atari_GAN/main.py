#!/usr/bin/env python
import random
import argparse
import cv2
import torch
import gym
import gym.spaces
import numpy as np
import GAN

BATCH_SIZE = 16
IMAGE_SIZE = 64 # dimension input image will be rescaled
REPORT_EVERY_ITER = 100
SAVE_IMAGE_EVERY_ITER = 1000

log = gym.logger
log.set_level(gym.logger.INFO)


class InputWrapper(gym.ObservationWrapper):
    """
    Preprocessing of input numpy array:
    1. resize image into predefined size
    2. move color channel axis to a first place
    """
    def __init__(self, *args):
        super(InputWrapper, self).__init__(*args)
        assert isinstance(self.observation_space, gym.spaces.Box)
        old_space = self.observation_space
        self.observation_space = gym.spaces.Box(self.observation(old_space.low), self.observation(old_space.high),
                                                dtype=np.float32)

    def observation(self, observation):
        # resize image
        new_obs = cv2.resize(observation, (IMAGE_SIZE, IMAGE_SIZE))
        # transform (210, 160, 3) -> (3, 210, 160)
        new_obs = np.moveaxis(new_obs, 2, 0)
        return new_obs.astype(np.float32) / 255.0


def iterate_batches(envs, batch_size=BATCH_SIZE):
    batch = [e.reset() for e in envs]
    env_gen = iter(lambda: random.choice(envs), None)

    while True:
        e = next(env_gen)
        obs, reward, is_done, _ = e.step(e.action_space.sample())
        if np.mean(obs) > 0.01:
            batch.append(obs)
        if len(batch) == batch_size:
            yield torch.tensor(np.array(batch, dtype=np.float32))
            batch.clear()
        if is_done:
            e.reset()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--cuda", default=False, action='store_true', help="Enable cuda computation")
    args = parser.parse_args()

    device = torch.device("cuda" if args.cuda else "cpu")
    print("Using device: "+ str(device))
    envs = [InputWrapper(gym.make(name)) for name in ('Breakout-v0', 'AirRaid-v0', 'Pong-v0')]
    input_shape = envs[0].observation_space.shape

    trainer = GAN.Trainer(input_shape, device)

    true_labels_v = torch.ones(BATCH_SIZE, dtype=torch.float32, device=device)
    fake_labels_v = torch.zeros(BATCH_SIZE, dtype=torch.float32, device=device)

    for batch_v in iterate_batches(envs):
        trainer.batch_v = batch_v.to(device)

        # generate extra fake samples, input is 4D: batch, filters, x, y
        gen_output_v = trainer.gen_fake(device)

        # train discriminator
        trainer.train_discr(gen_output_v, true_labels_v, fake_labels_v)

        # train generator
        trainer.train_gener(gen_output_v, true_labels_v)

        trainer.iter_no += 1

        if trainer.iter_no % REPORT_EVERY_ITER == 0:
            log.info("Iter %d: gen_loss=%.3e, dis_loss=%.3e", trainer.iter_no,
               np.mean(trainer.gen_losses), np.mean(trainer.dis_losses))
            trainer.write_summary()
        if trainer.iter_no % SAVE_IMAGE_EVERY_ITER == 0:
            trainer.save_images(gen_output_v)
