# RelLangSplat – Relationship Detection in 3DGS

## Pipeline ![pipeline](https://github.com/jackyhuuuu/RelLangSplat/raw/master/images/RelLangSplat_Pipeline.png)

## Spatial relation resolver ![spatial](https://github.com/jackyhuuuu/RelLangSplat/raw/master/images/Spatial_relation_resolver.png)

## Semantic relation resolver ![semantic](https://github.com/jackyhuuuu/RelLangSplat/raw/master/images/Semantic_relation_resolver.png)

# Overview
The model aims to deal with the relation query in the scene. Before using the model, you need to train your scene on LangSplat, which you can follow [LangSplat](https://github.com/minghanqin/LangSplat), the code to run LangSplat is also included in the repo.

Make sure your code have the following sturcture:
```RelLangSplat
<br>|---ckpts
<br>&nbsp;&nbsp;&nbsp;&nbsp;|---sam_vit_h_4b8939.pth
<br>|---lerf_ovs
<br>&nbsp;&nbsp;&nbsp;&nbsp;|---<scene_name>
<br>|---output
    

