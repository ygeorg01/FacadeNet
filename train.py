import data
import models
import optimizers
from options import TrainOptions
from util import IterationCounter
from util import Visualizer
from util import MetricTracker
from evaluation import GroupEvaluator
from torch.utils.tensorboard import SummaryWriter
import os
import shutil


opt = TrainOptions().parse()
dataset = data.create_dataset(opt)
opt.dataset = dataset

opt.phase = 'test'
eval_dataset = data.create_dataset(opt)
opt.eval_dataset = eval_dataset

opt.phase = 'train'

iter_counter = IterationCounter(opt)
visualizer = Visualizer(opt)
metric_tracker = MetricTracker(opt)
evaluators = GroupEvaluator(opt)

model = models.create_model(opt)
optimizer = optimizers.create_optimizer(opt, model)

print('Opt name: ', opt.name)

# Visualization
if os.path.exists("runs/" + opt.name):
    shutil.rmtree("runs/" + opt.name)

writer = SummaryWriter("runs/" + opt.name)

while not iter_counter.completed_training():
    with iter_counter.time_measurement("data"):
        cur_data = next(dataset)

    with iter_counter.time_measurement("train"):

        losses, real_D_pred, rec_D_pred = optimizer.train_one_step(cur_data, iter_counter.steps_so_far)

        metric_tracker.update_metrics(losses, smoothe=True)

    with iter_counter.time_measurement("maintenance"):
        if iter_counter.needs_printing():
            visualizer.print_current_losses(iter_counter.steps_so_far,
                                            iter_counter.time_measurements,
                                            metric_tracker.current_metrics())

            for loss_name, loss_value in metric_tracker.current_metrics().items():
                writer.add_scalar('Loss/'+loss_name, loss_value, iter_counter.steps_so_far)
                # if real_D_pred is not None:
                #     print(real_D_pred, real_D_pred.view(-1))
                #     import numpy as np
                #     writer.add_histogram('DistriminHistogram/ ', np.array([0.5, 0.2, 0.3]), iter_counter.steps_so_far)
                #     writer.add_histogram('Distriminator Histogram/ ', rec_D_pred, iter_counter.steps_so_far)


            # if real_pred is not None:
            #     writer.add_histogram("D/real_pred", real_pred, iter_counter.steps_so_far)
            #     writer.add_histogram("D/rec_pred", rec_pred, iter_counter.steps_so_far)

        if iter_counter.needs_displaying():
            visuals = optimizer.get_visuals_for_snapshot(cur_data)
            visualizer.display_current_results(visuals,
                                               iter_counter.steps_so_far)

        if iter_counter.needs_evaluation():
            metrics = evaluators.evaluate(
                model, eval_dataset, iter_counter.steps_so_far)

            # metrics = evaluators.evaluate_metrics(model, eval_dataset)
            # print(metrics)
            # Eval metrics removed
            writer.add_scalar('Evaluation/psnr', metrics['psnr'], iter_counter.steps_so_far)
            writer.add_scalar('Evaluation/ssim', metrics['ssim'], iter_counter.steps_so_far)
            writer.add_scalar('Evaluation/fid', metrics['fid'], iter_counter.steps_so_far)
            writer.add_scalar('Evaluation/fid_swap', metrics['fid_swap'], iter_counter.steps_so_far)
            writer.add_scalar('Evaluation/LPIPS_vgg', metrics['LPIPS_vgg'], iter_counter.steps_so_far)
            writer.add_scalar('Evaluation/LPIPS_alex', metrics['LPIPS_alex'], iter_counter.steps_so_far)

        if iter_counter.needs_saving():
            optimizer.save(iter_counter.steps_so_far)

        if iter_counter.completed_training():
            break

        # break

        iter_counter.record_one_iteration()

optimizer.save(iter_counter.steps_so_far)
print('Training finished.')
