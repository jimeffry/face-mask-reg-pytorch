#!/usr/bin/bash
#bm_mafa_best1
#bm_mafa_best1_1
#bm_mafa_best1_2 backone resnet class is 5, training data include 7 videos，roc=0.928,p=0.994,fpr=0.0009
#bm_mafa_best1_3 backone reset class is 6, training data include 1 video=0.93
#bm_mafa_best1_4 backone resnet+ feature maps+seg maps class is 2, training data include 4 videos
#bm_mafa_best1_5 backone resnet class is 5, training data include 4 videos
#bm_mafa_best5_1 backone resnet +2 fc, class is 5, training data include 6videos,1:roc=939,p=0.993,fpr=0.0011
#bm_mafa_best5_2 backone resnet +2 fc, class is 5, training data include 7videos,1:roc=923,p=0.991,fpr=0.0015
#*******************
## train net CUDA_VISIBLE_DEVICES=0,1
# python train/train.py --lr 0.0004 --dataset mafa --cuda false --save_folder /data/models/breathmask/ --batch_size 2 --multigpu false  #--resume /data/models/head/sfd_head_60000.pth

##*************  /data/detect/shang_crowed/part_A_final/test_data/test_data.txt
# python test/demo.py --file_in /home/lxy/Develop/git_prj/BaiduImageSpider/wear_mask/46.jpg --breath_modelpath /data/models/breathmask/bm_mafa_best_1.pth --detect-model-dir ../models/face_detect_models
# python test/demo.py --file_in video --breath_modelpath /data/models/breathmask/bm_mafa_best1_2.pth --detect-model-dir ../models/face_detect_models
# python test/demo.py --file_in video --breath_modelpath /data/models/breathmask/bm_mafa_test2.pth --detect-model-dir ../models/face_detect_models
# python test/demo.py --file_in /data/t31.mp4 --breath_modelpath /data/models/breathmask/bm_mafa_best1_2.pth --detect-model-dir ../models/face_detect_models --img-dir /data/detect/breathmask/data_collect
# python test/demo.py --file_in /data/masktest/ --breath_modelpath /data/models/breathmask/bm_mafa_best1_2.pth --detect-model-dir ../models/face_detect_models --save-dir /data/maskresult
 #*********************retinaface
#  python test/demo.py --file_in /data/t31.mp4 --breath_modelpath /data/models/breathmask/bm_mafa_test2.pth --detect_modelpath /data/models/retinaface/retinaface_epoch_120.pb --img-dir /data/detect/breathmask/data_collect
#  python test/demo.py --file_in video --breath_modelpath /data/models/breathmask/bm_mafa_test2.pth --detect_modelpath /data/models/retinaface/retinaface_epoch_120.pb 
#   python test/demo.py --file_in /data/masktest/ --breath_modelpath /data/models/breathmask/bm_mafa_best1_2.pth --detect_modelpath /data/models/retinaface/retinaface_epoch_120.pb  --save-dir /data/maskresult2
#*****************tensorflow
# python test/demo.py --file_in video --breath_modelpath /data/models/breathmask/mask.pb --detect-model-dir ../models/face_detect_models
# python test/demo.py --file_in /home/lxy/Develop/git_prj/BaiduImageSpider/wear_mask/80.jpg --breath_modelpath ../models/breathmaskv1-1.pb --detect-model-dir ../models/face_detect_models
# python test/demo.py --file_in /data/b11.mp4 --breath_modelpath ../models/bm_tf2.pb --detect-model-dir ../models/face_detect_models
#*************************test
# python test/eval.py --breath_modelpath /data/models/breathmask/bm_mafa_best5_2.pth  --file_in ./breathmask_test.txt --out_file ../data/breathmask_resnetv2_6.txt --img_dir /data/detect/breathmask/images --save_dir /data/detect/breathmask/test_result_resnetv26
# python test/plot.py --file-in ../data/breathmaskv3.txt --base-name breathmaskv2 --cmd-type plot3data

#**convert model
# python test/tr2tf.py

#********************utils
python utils/scripts.py
#*************************testcar
#  python test/eval.py --breath_modelpath /data/models/rubbishcar/rbcar_mafa_best.pth  --file_in ../data/rubbish_val.txt --out_file ../data/rubbish_resnet_testr.txt --img_dir /data/detect/zhatu_car --save_dir /data/detect/zhatu_car/test_result_resnet



#*********************face attribute demo
# python test/demo_attr.py --file_in video --faceattr_modelpath /data/models/face_attribute/CelebA/faceattr_resnet50.pb --detect-model-dir ../models/face_detect_models
# python test/eval_faceattr.py --file-in ../data/test_faceattr.txt --base-dir /data/Face_Reg/CelebA/img_detected --out-file ../data/faceattr_result.txt  --record-file ../data/faceattr.csv  --faceattr_modelpath /data/models/face_attribute/CelebA/faceattr_resnet50.pb --cmd-type evalue
# python utils/plot_vis.py --file-in ../data/faceattr.csv --basename faceattr_vis
# python utils/plot.py --file-in ../data/faceattr_result.txt --base-name faceattr2 --cmd-type plot3data
# python test/demo_attr.py --file_in /data/detect/faceattribute --faceattr_modelpath /data/models/face_attribute/CelebA/faceattr_resnet50.pb --detect-model-dir ../models/face_detect_models --save_dir /data/detect/faceresult


#********************test unsell
#unsell2resnet_siping_best.pth backbone is resnet50 and class num is 2. total test num 468, right num 445
#unsell_siping_best.pth backbone is resnet50_cbam and class num is 5. total test num 468, right num 444
#unsell2_siping_best.pth backone is resnet50_cbam and class num is 2. total test num 468, right num 446
#unsell7_siping_best.pth backone is resnet50_cbam and class num is 7. total test num 468, right num 448
# python test/eval_unsell.py --file-in ../data/unsell_val.txt --base-dir /data/detect/siping_caiji --out-file ../data/unsellresnet18_result.txt  --record-file ../data/unsellresnet18_val.csv  --unsell_modelpath /data/models/unsell_cls/unsell2resnet18_siping_best.pth --cmd-type evalue
# python utils/plot_vis.py --file-in ../data/unsellresnet18_val.csv --basename unsellresnet18_vis
# python utils/plot.py --file-in ../data/unsellresnet18_result.txt --base-name unsellresnet18_pr --cmd-type plot3data

# python test/demo_unsell.py --file_in ../data/unsell_tobedone.txt --unsell_modelpath /data/models/unsell_cls/unsell2resnet_siping_best.pth  --img-dir /data/detect/siping_caiji/cls_done/bg_imgs --save-dir /data/detect/siping_caiji/cls_tofine
# python test/demo_unsell.py --file_in ../data/unsell_val.txt --unsell_modelpath /data/models/unsell_cls/unsell2resnet18_siping_bestv2.pth  --img-dir /data/detect/siping_caiji --save-dir /data/detect/siping_caiji/unsell2resnet18_val_test
# python test/demo_unsell.py --file_in /data/detect/siping_caiji/images_done/fg --unsell_modelpath /data/models/unsell_cls/unsell2resnet18_siping_bestv2.pth   
#***tensorflow
# python test/demo_unsell.py --file_in  /data/detect/siping_caiji/images_done/fg --unsell_modelpath ../models/unsell_resnet18v2.pb --img-dir /data/detect/siping_caiji #--save-dir /data/detect/siping_caiji/unsellres18_val_test_tf
# python test/demo_unsell.py --file_in /data/detect/siping_caiji/images_done/fg/22030361001321400354_72665_2_3-0.3538.jpg --unsell_modelpath ../models/unsell_resnet18.pb

#****************************** test smoking calling
# python test/demo_unsell.py --file_in ../data/ehualu_test.txt --unsell_modelpath /data/models/unsell_cls/sc_ehualu_best.pth  --img-dir /data/videos/ehualu/test
# python test/load_json.py