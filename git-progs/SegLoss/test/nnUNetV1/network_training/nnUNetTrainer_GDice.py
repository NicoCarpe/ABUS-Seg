from nnunet.training.loss_functions.dice_loss import GDiceLossV2
# from nnunet.training.loss_functions.dice_loss import GDiceLoss
from nnunet.training.network_training.nnUNetTrainer import nnUNetTrainer
from nnunet.utilities.nd_softmax import softmax_helper


class nnUNetTrainer_GDice(nnUNetTrainer):
    def __init__(self, plans_file, fold, output_folder=None, dataset_directory=None, batch_dice=True, stage=None,
                 unpack_data=True, deterministic=True, fp16=False):
        super().__init__(plans_file, fold, output_folder, dataset_directory, batch_dice, stage,
                                              unpack_data, deterministic, fp16)
        self.apply_nonlin = softmax_helper
        self.loss = GDiceLossV2(apply_nonlin=self.apply_nonlin, smooth=1e-5)
