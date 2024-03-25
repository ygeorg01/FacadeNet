from .tmux_launcher import Options, TmuxLauncher


class Launcher(TmuxLauncher):
    def options(self):
        opt = Options()
        opt.set(

            dataroot="/media/yiangos/Urban_Enviroment_Understanding_Project/LSAA/facade_data_full/interactive_textures_sem/facades/",
            dataroot_sem="/media/yiangos/Urban_Enviroment_Understanding_Project/LSAA/facade_data_full/interactive_textures_sem/semantics/",
            dataroot_feats="/media/yiangos/Urban_Enviroment_Understanding_Project/LSAA/facade_data_full/interactive_textures_sem/vit_feats_4/",
            dataroot_depth="/media/yiangos/Urban_Enviroment_Understanding_Project/LSAA/facade_data_full/interactive_textures_sem/depth/",
            dataroot_h="/media/yiangos/Urban_Enviroment_Understanding_Project/LSAA/facade_data_full/interactive_textures_sem/horizontal_maps/",
            dataroot_v="/media/yiangos/Urban_Enviroment_Understanding_Project/LSAA/facade_data_full/interactive_textures_sem/vertical_maps/",
            train_split_dir="/media/yiangos/Urban_Enviroment_Understanding_Project/LSAA/facade_data_full/interactive_textures_sem/train_split.txt",
            eval_split_dir ="/media/yiangos/Urban_Enviroment_Understanding_Project/LSAA/facade_data_full/interactive_textures_sem/eval_split.txt",

            #dataroot="/lustreFS/data/vcg/yiangos/interactive_textures_sem/facades/",
            #dataroot_feats="/lustreFS/data/vcg/yiangos/interactive_textures_sem/vit_feats_4/",
            #dataroot_depth="/lustreFS/data/vcg/yiangos/interactive_textures_sem/depth/",
            #dataroot_h="/lustreFS/data/vcg/yiangos/interactive_textures_sem/horizontal_maps/",
            #dataroot_v="/lustreFS/data/vcg/yiangos/interactive_textures_sem/vertical_maps/",
            #train_split_dir="/lustreFS/data/vcg/yiangos/interactive_textures_sem/train_split.txt",
            #eval_split_dir="/lustreFS/data/vcg/yiangos/interactive_textures_sem/eval_split.txt",

            dataset_mode="imagefolder_va_depth_lr",
            num_gpus=1, batch_size=2,
            preprocess="resize_and_crop",
            nThreads=6,
            # scale the image such that the short side is |load_size|, and
            # crop a square window of |crop_size|.
            load_size=300, crop_size=256,
            display_freq=20000, print_freq=1000,
            model='autoencoder_va_vec_lr', optimizer='autoencoder_va_vec_lr',
            lambda_L1=3,
            checkpoints_dir="./checkpoints/",
            lambda_sel_L1=3
            # parser.add_argument('--model', type=str, default='swapping_autoencoder', help='which model to use')
            # parser.add_argument('--optimizer', type=str, default='swapping_autoencoder', help='which model to use')
        ),

        return [
            opt.specify(
                name="facadenet",
            ),
        ]

    def train_options(self):
        common_options = self.options()
        return [opt.specify(
            continue_train=False,
            evaluation_metrics="va_visualization_vec",
            evaluation_freq=250000,
            #evaluation_freq=5000
        ) for opt in common_options]

    def test_options_fid(self):
        return []

    def test_options(self):
        common_options = self.options()
        root = "/media/yiangos/Urban_Enviroment_Understanding_Project/LSAA/facade_data_full/interactive_textures_sem/"
        return [opt.tag("fig4").specify(
            load_size=256,
            crop_size=256,
            phase='test',
            num_gpus=1,
            batch_size=4,
            dataroot=root+"facades/",
            dataroot_sem= root+"semantics/",
            dataroot_feats= root+"vit_feats_4/",
            dataroot_depth= root+"depth/",
            dataroot_h= root+"horizontal_maps/",
            dataroot_v= root+"vertical_maps/",
            train_split_dir= root+"train_split.txt",
            eval_split_dir = root+"eval_split.txt",
            dataset_mode="imagefolder_va_depth_lr",
            preprocess="resize",
            evaluation_metrics="va_visualization_vec",
            generator_name="facadenet.pth"
        ) for opt in common_options]

