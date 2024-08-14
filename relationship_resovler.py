import os
import json
import torch
from argparse import ArgumentParser
from arguments import PipelineParams
from text_prompt import seed_everything, evaluate_with_prompt
from relationship.word import sentence_decompose, relationship_analysis, relationship_match, relationship_detect
from relationship.semantic import SemanticResolver
from query_3d import LangSplat_scene
from multi_view_gen import camera_pose_sample
from render_custom_view import render_custom_view
from PIL import Image


seed_num = 42
seed_everything(seed_num)
parser = ArgumentParser(description="prompt any relationship query")
parser.add_argument('--encoder_dims',
                    nargs='+',
                    type=int,
                    default=[256, 128, 64, 32, 3],
                    )
parser.add_argument('--decoder_dims',
                    nargs='+',
                    type=int,
                    default=[16, 32, 64, 128, 256, 256, 512],
                    )

pipeline = PipelineParams(parser)
parser.add_argument("--iteration", default=-1, type=int)
parser.add_argument("--skip_train", action="store_true")
parser.add_argument("--skip_test", action="store_true")
parser.add_argument("--quiet", action="store_true")
parser.add_argument("--include_feature", action="store_true")
parser.add_argument("--dataset_name", type=str, default="teatime")
parser.add_argument('--feat_dir', type=str, default="output")
parser.add_argument("--ae_ckpt_dir", type=str, default="autoencoder/ckpt")
parser.add_argument("--output_dir", type=str, default="eval_result")
parser.add_argument("--mask_thresh", type=float, default=0.4)
parser.add_argument("--ply_path", type=str, default="output/teatime_3/point_cloud/iteration_30000/point_cloud.ply")
parser.add_argument("--feature_path", type=str, default="lerf_ovs/teatime/teatime_3_features.npy")

args = parser.parse_args()
dataset_name = args.dataset_name
mask_thresh = args.mask_thresh
feat_dir = [os.path.join(args.feat_dir, dataset_name + f"_{i}", "train/ours_None/renders_npy") for i in range(1, 4) if
            i != 2]
output_path = os.path.join(args.output_dir, dataset_name)
ae_ckpt_path = os.path.join(args.ae_ckpt_dir, dataset_name, "best_ckpt.pth")
camera_pose_path = "lerf_ovs/teatime/output/teatime/cameras.json"
custom_camera_pose_path = "custom_view/teatime_3/cameras.json"
custom_view_img_path = "custom_view/teatime_3/rgb/ours_None/renders"

####################################################################################################
query_image = "lerf_ovs/teatime/images/frame_00050.jpg"
query = "A white sheep drinking a glass of tea"
sub, obj, rel = sentence_decompose(query)
rel_type = relationship_analysis(rel)

print(f"Subject: {sub}")
print(f"Object: {obj}")
print(f"Relationship: {rel}")
print(f"Relationship type: {rel_type}")

if rel_type == "spatial":
    box_sub = evaluate_with_prompt([sub], query_image, feat_dir, output_path, ae_ckpt_path, mask_thresh,
                                   args.encoder_dims, args.decoder_dims)
    box_obj = evaluate_with_prompt([obj], query_image, feat_dir, output_path, ae_ckpt_path, mask_thresh,
                                   args.encoder_dims, args.decoder_dims)
    if box_sub == None or box_obj == None:
        if box_sub == None and box_obj != None:
            print(f"Invalid query, because can't find {sub} in the scene!")
        elif box_sub != None and box_obj == None:
            print(f"Invalid query, because can't find {obj} in the scene!")
        else:
            print(f"Invalid query, because can't find {sub} and {obj} in the scene!")
    else:
        rel = relationship_match(rel)
        print(f"Relationship: {rel}")
        output = relationship_detect(query, rel, box_sub, box_obj)
        print(f"{output}")

elif rel_type == "semantic":
    LangSplat_model = LangSplat_scene(args.ply_path, args.feature_path, ae_ckpt_path, args.encoder_dims,
                                      args.decoder_dims)
    sub_center = LangSplat_model.query_3d(sub)
    obj_center = LangSplat_model.query_3d(obj)

    if not torch.all(sub_center == 0) and not torch.all(obj_center == 0):
        with open(camera_pose_path, 'r') as file:
            cam_json = json.load(file)

        camera_pose = camera_pose_sample(sub_center, cam_json)
        update_cam_json = [cam_json[idx] for idx in camera_pose]

        with open(custom_camera_pose_path, 'w') as file:
            json.dump(update_cam_json, file)

        render_custom_view(pipeline, args)
        image_paths = []
        for root, dirs, files in os.walk(custom_view_img_path):
            for file in files:
                image_paths.append(os.path.join(root, file))

        semantic_resolver = SemanticResolver()
        image = Image.open(query_image)
        output = semantic_resolver.evaluate_semantic_relationship(image, query)
        # decide if the output is positive
        for img in image_paths:
            image = Image.open(img)
            output = semantic_resolver.evaluate_semantic_relationship(image, query)
            # decide if the output is positive
            print(f"output: {output}")
            break
    else:
        if torch.all(sub_center == 0) and not torch.all(obj_center == 0):
            print(f"Invalid query, because can find {sub} in the scene!")
        elif torch.all(obj_center == 0) and not torch.all(sub_center == 0):
            print(f"Invalid query, because can find {obj} in the scene!")
        else:
            print(f"Invalid query, because can find {sub} and {obj} in the scene!")

