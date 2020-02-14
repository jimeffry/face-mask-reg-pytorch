#!/usr/bin/bash
#*******************
## train net CUDA_VISIBLE_DEVICES=0,1
# python train/train.py --lr 0.0004 --dataset mafa --cuda false --save_folder /data/models/breathmask/ --batch_size 2 --multigpu false  #--resume /data/models/head/sfd_head_60000.pth

## demo  /data/detect/shang_crowed/part_A_final/test_data/test_data.txt
# python test/demo.py --file_in /home/lxy/Develop/git_prj/BaiduImageSpider/wear_mask/46.jpg --breath_modelpath /data/models/breathmask/bm_mafa_best_1.pth --detect-model-dir ../models/face_detect_models
# python test/demo.py --file_in video --breath_modelpath /data/models/breathmask/bm_mafa_best.pth --detect-model-dir ../models/face_detect_models
# python test/demo.py --file_in video --breath_modelpath /data/models/breathmask/mask.pb --detect-model-dir ../models/face_detect_models
# python test/demo.py --file_in /home/lxy/Develop/git_prj/BaiduImageSpider/wear_mask/80.jpg --breath_modelpath ../models/breathmaskv1-1.pb --detect-model-dir ../models/face_detect_models
# python test/demo.py --file_in /home/lxy/Develop/git_prj/BaiduImageSpider/wear_mask/80.jpg --breath_modelpath /data/models/breathmask/mask.pb --detect-model-dir ../models/face_detect_models
#*************************test
python test/eval.py --breath_modelpath /data/models/breathmask/bm_mafa_best.pth  --file_in ../data/mafa_celeba_test.txt --out_file ../data/breathmask_resnet2.txt --img_dir /data/Face_Reg/CelebA/img_detected --save_dir /data/detect/MAFA/test_result_resnet2
# python test/plot.py --file-in ../data/breathmaskv1.txt --base-name breathmask --cmd-type plot3data

#**convert model
# python test/tr2tf.py
