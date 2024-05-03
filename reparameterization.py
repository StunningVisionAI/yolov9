import torch
import argparse
from models.yolo import Model

# Add Parser
parser = argparse.ArgumentParser()
   
parser.add_argument("--weights", type=str, default="yolov9-c.pt", help="YOLOv9 Weights")
parser.add_argument("--variant", type=str, default="c", help="YOLOv9 Variant")
parser.add_argument("--classes", type=int, default=80, help="Number of Classes")
parser.add_argument("--output", type=str, default="yolov9-c.pt", help="Output Filename")

args = parser.parse_args()

def reparameter_c_variant():
    idx = 0
    for k, v in model.state_dict().items():
        if "model.{}.".format(idx) in k:
            if idx < 22:
                kr = k.replace("model.{}.".format(idx), "model.{}.".format(idx+1))
                model.state_dict()[k] -= model.state_dict()[k]
                model.state_dict()[k] += ckpt['model'].state_dict()[kr]
            elif "model.{}.cv2.".format(idx) in k:
                kr = k.replace("model.{}.cv2.".format(idx), "model.{}.cv4.".format(idx+16))
                model.state_dict()[k] -= model.state_dict()[k]
                model.state_dict()[k] += ckpt['model'].state_dict()[kr]
            elif "model.{}.cv3.".format(idx) in k:
                kr = k.replace("model.{}.cv3.".format(idx), "model.{}.cv5.".format(idx+16))
                model.state_dict()[k] -= model.state_dict()[k]
                model.state_dict()[k] += ckpt['model'].state_dict()[kr]
            elif "model.{}.dfl.".format(idx) in k:
                kr = k.replace("model.{}.dfl.".format(idx), "model.{}.dfl2.".format(idx+16))
                model.state_dict()[k] -= model.state_dict()[k]
                model.state_dict()[k] += ckpt['model'].state_dict()[kr]
        else:
            while True:
                idx += 1
                if "model.{}.".format(idx) in k:
                    break
            if idx < 22:
                kr = k.replace("model.{}.".format(idx), "model.{}.".format(idx+1))
                model.state_dict()[k] -= model.state_dict()[k]
                model.state_dict()[k] += ckpt['model'].state_dict()[kr]
            elif "model.{}.cv2.".format(idx) in k:
                kr = k.replace("model.{}.cv2.".format(idx), "model.{}.cv4.".format(idx+16))
                model.state_dict()[k] -= model.state_dict()[k]
                model.state_dict()[k] += ckpt['model'].state_dict()[kr]
            elif "model.{}.cv3.".format(idx) in k:
                kr = k.replace("model.{}.cv3.".format(idx), "model.{}.cv5.".format(idx+16))
                model.state_dict()[k] -= model.state_dict()[k]
                model.state_dict()[k] += ckpt['model'].state_dict()[kr]
            elif "model.{}.dfl.".format(idx) in k:
                kr = k.replace("model.{}.dfl.".format(idx), "model.{}.dfl2.".format(idx+16))
                model.state_dict()[k] -= model.state_dict()[k]
                model.state_dict()[k] += ckpt['model'].state_dict()[kr]
    _ = model.eval()


def reparameter_e_variant():
    idx = 0
    for k, v in model.state_dict().items():
        if "model.{}.".format(idx) in k:
            if idx < 29:
                kr = k.replace("model.{}.".format(idx), "model.{}.".format(idx))
                model.state_dict()[k] -= model.state_dict()[k]
                model.state_dict()[k] += ckpt['model'].state_dict()[kr]
                print(k, "perfectly matched!!")
            elif idx < 42:
                kr = k.replace("model.{}.".format(idx), "model.{}.".format(idx+7))
                model.state_dict()[k] -= model.state_dict()[k]
                model.state_dict()[k] += ckpt['model'].state_dict()[kr]
                print(k, "perfectly matched!!")
            elif "model.{}.cv2.".format(idx) in k:
                kr = k.replace("model.{}.cv2.".format(idx), "model.{}.cv4.".format(idx+7))
                model.state_dict()[k] -= model.state_dict()[k]
                model.state_dict()[k] += ckpt['model'].state_dict()[kr]
                print(k, "perfectly matched!!")
            elif "model.{}.cv3.".format(idx) in k:
                kr = k.replace("model.{}.cv3.".format(idx), "model.{}.cv5.".format(idx+7))
                model.state_dict()[k] -= model.state_dict()[k]
                model.state_dict()[k] += ckpt['model'].state_dict()[kr]
                print(k, "perfectly matched!!")
            elif "model.{}.dfl.".format(idx) in k:
                kr = k.replace("model.{}.dfl.".format(idx), "model.{}.dfl2.".format(idx+7))
                model.state_dict()[k] -= model.state_dict()[k]
                model.state_dict()[k] += ckpt['model'].state_dict()[kr]
                print(k, "perfectly matched!!")
        else:
            while True:
                idx += 1
                if "model.{}.".format(idx) in k:
                    break
            if idx < 29:
                kr = k.replace("model.{}.".format(idx), "model.{}.".format(idx))
                model.state_dict()[k] -= model.state_dict()[k]
                model.state_dict()[k] += ckpt['model'].state_dict()[kr]
                print(k, "perfectly matched!!")
            elif idx < 42:
                kr = k.replace("model.{}.".format(idx), "model.{}.".format(idx+7))
                model.state_dict()[k] -= model.state_dict()[k]
                model.state_dict()[k] += ckpt['model'].state_dict()[kr]
                print(k, "perfectly matched!!")
            elif "model.{}.cv2.".format(idx) in k:
                kr = k.replace("model.{}.cv2.".format(idx), "model.{}.cv4.".format(idx+7))
                model.state_dict()[k] -= model.state_dict()[k]
                model.state_dict()[k] += ckpt['model'].state_dict()[kr]
                print(k, "perfectly matched!!")
            elif "model.{}.cv3.".format(idx) in k:
                kr = k.replace("model.{}.cv3.".format(idx), "model.{}.cv5.".format(idx+7))
                model.state_dict()[k] -= model.state_dict()[k]
                model.state_dict()[k] += ckpt['model'].state_dict()[kr]
                print(k, "perfectly matched!!")
            elif "model.{}.dfl.".format(idx) in k:
                kr = k.replace("model.{}.dfl.".format(idx), "model.{}.dfl2.".format(idx+7))
                model.state_dict()[k] -= model.state_dict()[k]
                model.state_dict()[k] += ckpt['model'].state_dict()[kr]
                print(k, "perfectly matched!!")
    _ = model.eval()

device = torch.device("cpu")
yolov9_variant = args.variant

cfg = "models\detect\gelan-c.yaml"
if(yolov9_variant == "e"):
    cfg = "models\detect\gelan-e.yaml"

model = Model(cfg, ch=3, nc=args.classes, anchors=3)
#model = model.half()
model = model.to(device)
_ = model.eval()
ckpt = torch.load(args.weights, map_location='cpu')
model.names = ckpt['model'].names
model.nc = ckpt['model'].nc

if(yolov9_variant == "e"):
    reparameter_e_variant()
else:
    reparameter_c_variant()

output_filename = args.output + ".pt"

m_ckpt = {'model': model.half(),
          'optimizer': None,
          'best_fitness': None,
          'ema': None,
          'updates': None,
          'opt': None,
          'git': None,
          'date': None,
          'epoch': -1}
torch.save(m_ckpt, output_filename)

print("Completed")