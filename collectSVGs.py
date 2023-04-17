import yaml
import argparse
import numpy as np
import os

from models import *
from experiment import VAEXperiment
import torch.backends.cudnn as cudnn
from pytorch_lightning import Trainer
from pytorch_lightning.loggers import TestTubeLogger
from pytorch_lightning.callbacks import ModelCheckpoint
import matplotlib.pyplot as plt

import torch
import pydiffvg
import xml.etree.ElementTree as etree
from xml.dom import minidom

def make_tensor(x, grad=False):
    x = torch.tensor(x, dtype=torch.float32)
    x.requires_grad = grad
    return x

def raster(curves, all_points, color=[0,0,0, 1], verbose=False, white_background=True):
        assert len(color) == 4
        # print('1:', process.memory_info().rss*1e-6)
        render_size = 128
        bs = all_points.shape[0]
        if verbose:
            render_size = render_size*2
        outputs = []
        all_points = all_points*render_size
        num_ctrl_pts = torch.zeros(curves, dtype=torch.int32).to(all_points.device) + 2
        color = make_tensor(color).to(all_points.device)
        for k in range(bs):
            # Get point parameters from network
            render = pydiffvg.RenderFunction.apply
            shapes = []
            shape_groups = []
            points = all_points[k].contiguous()#[self.sort_idx[k]] # .cpu()

            if verbose:
                np.random.seed(0)
                colors = np.random.rand(curves, 4)
                high = np.array((0.565, 0.392, 0.173, 1))
                low = np.array((0.094, 0.310, 0.635, 1))
                diff = (high-low)/(curves)
                colors[:, 3] = 1
                for i in range(curves):
                    scale = diff*i
                    color = low + scale
                    color[3] = 1
                    color = torch.tensor(color)
                    num_ctrl_pts = torch.zeros(1, dtype=torch.int32) + 2
                    if i*3 + 4 > curves * 3:
                        curve_points = torch.stack([points[i*3], points[i*3+1], points[i*3+2], points[0]])
                    else:
                        curve_points = points[i*3:i*3 + 4]
                    path = pydiffvg.Path(
                        num_control_points=num_ctrl_pts, points=curve_points,
                        is_closed=False, stroke_width=torch.tensor(4))
                    path_group = pydiffvg.ShapeGroup(
                        shape_ids=torch.tensor([i]),
                        fill_color=None,
                        stroke_color=color)
                    shapes.append(path)
                    shape_groups.append(path_group)
                # for i in range(self.curves * 3):
                #     scale = diff*(i//3)
                #     color = low + scale
                #     color[3] = 1
                #     color = torch.tensor(color)
                #     if i%3==0:
                #         # color = torch.tensor(colors[i//3]) #green
                #         shape = pydiffvg.Rect(p_min = points[i]-8,
                #                              p_max = points[i]+8)
                #         group = pydiffvg.ShapeGroup(shape_ids=torch.tensor([self.curves+i]),
                #                                            fill_color=color)
                #
                #     else:
                #         # color = torch.tensor(colors[i//3]) #purple
                #         shape = pydiffvg.Circle(radius=torch.tensor(8.0),
                #                                  center=points[i])
                #         group = pydiffvg.ShapeGroup(shape_ids=torch.tensor([self.curves+i]),
                #                                            fill_color=color)
                #     shapes.append(shape)
                #     shape_groups.append(group)

            else:

                path = pydiffvg.Path(
                    num_control_points=num_ctrl_pts, points=points,
                    is_closed=True)

                shapes.append(path)
                path_group = pydiffvg.ShapeGroup(
                    shape_ids=torch.tensor([len(shapes) - 1]),
                    fill_color=color,
                    stroke_color=color)
                shape_groups.append(path_group)

            return shapes, shape_groups
            
scale = 1
def decode_and_composite(self, z: Tensor, save_dir, name, return_points=False,**kwargs):
    bs = z.shape[0]
    layers = []
    save_all_points = []
    n = len(self.colors)
    loss = 0
    z_rnn_input = z[None, :, :].repeat(n, 1, 1)  # [len, batch size, emb dim]
    outputs, hidden = self.rnn(z_rnn_input)
    outputs = outputs.permute(1, 0, 2)  # [batch size, len, emb dim]
    outputs = outputs[:, :, :self.latent_dim] + outputs[:, :, self.latent_dim:]
    z_layers = []
    shapes = []
    shape_groups = []
    for i in range(n):
        shape_output = self.divide_shape(outputs[:, i, :])
        shape_latent = self.final_shape_latent(shape_output)
        all_points = self.decode(shape_latent)#, point_predictor=self.point_predictor[i])
        save_all_points.append(all_points)
        shape, shape_group = raster(self.curves, all_points*scale, self.colors[i], verbose=kwargs['verbose'], white_background=False)
        shapes.append(shape)
        shape_groups.append(shape_group)

    print(f"Saving {save_dir}VectorVAEnLayers/{version_dir}/svgs/{name}.svg")
    save_svg(f"{save_dir}VectorVAEnLayers/{version_dir}/svgs/{name}.svg", self.imsize, self.imsize, shapes, shape_groups)    

def prettify(elem):
    """Return a pretty-printed XML string for the Element.
    """
    rough_string = etree.tostring(elem, 'utf-8')
    reparsed = minidom.parseString(rough_string)
    return reparsed.toprettyxml(indent="  ")

def save_svg(filename, width, height, shapes_list, shape_groups_list, use_gamma = False):
    root = etree.Element('svg')
    root.set('version', '1.1')
    root.set('xmlns', 'http://www.w3.org/2000/svg')
    root.set('width', str(width))
    root.set('height', str(height))
    defs = etree.SubElement(root, 'defs')
    g = etree.SubElement(root, 'g')
    if use_gamma:
        f = etree.SubElement(defs, 'filter')
        f.set('id', 'gamma')
        f.set('x', '0')
        f.set('y', '0')
        f.set('width', '100%')
        f.set('height', '100%')
        gamma = etree.SubElement(f, 'feComponentTransfer')
        gamma.set('color-interpolation-filters', 'sRGB')
        feFuncR = etree.SubElement(gamma, 'feFuncR')
        feFuncR.set('type', 'gamma')
        feFuncR.set('amplitude', str(1))
        feFuncR.set('exponent', str(1/2.2))
        feFuncG = etree.SubElement(gamma, 'feFuncG')
        feFuncG.set('type', 'gamma')
        feFuncG.set('amplitude', str(1))
        feFuncG.set('exponent', str(1/2.2))
        feFuncB = etree.SubElement(gamma, 'feFuncB')
        feFuncB.set('type', 'gamma')
        feFuncB.set('amplitude', str(1))
        feFuncB.set('exponent', str(1/2.2))
        feFuncA = etree.SubElement(gamma, 'feFuncA')
        feFuncA.set('type', 'gamma')
        feFuncA.set('amplitude', str(1))
        feFuncA.set('exponent', str(1/2.2))
        g.set('style', 'filter:url(#gamma)')

    # Store color
    for shapes, shape_groups in zip(shapes_list, shape_groups_list):
        for i, shape_group in enumerate(shape_groups):
            def add_color(shape_color, name):
                if isinstance(shape_color, pydiffvg.LinearGradient):
                    lg = shape_color
                    color = etree.SubElement(defs, 'linearGradient')
                    color.set('id', name)
                    color.set('x1', str(lg.begin[0].item()))
                    color.set('y1', str(lg.begin[1].item()))
                    color.set('x2', str(lg.end[0].item()))
                    color.set('y2', str(lg.end[1].item()))
                    offsets = lg.offsets.data.cpu().numpy()
                    stop_colors = lg.stop_colors.data.cpu().numpy()
                    for j in range(offsets.shape[0]):
                        stop = etree.SubElement(color, 'stop')
                        stop.set('offset', offsets[j])
                        c = lg.stop_colors[j, :]
                        stop.set('stop-color', 'rgb({}, {}, {})'.format(\
                            int(255 * c[0]), int(255 * c[1]), int(255 * c[2])))
                        stop.set('stop-opacity', '{}'.format(c[3]))

            if shape_group.fill_color is not None:
                add_color(shape_group.fill_color, 'shape_{}_fill'.format(i))
            if shape_group.stroke_color is not None:
                add_color(shape_group.stroke_color, 'shape_{}_stroke'.format(i))

        for i, shape_group in enumerate(shape_groups):
            shape = shapes[shape_group.shape_ids[0]]
            if isinstance(shape, pydiffvg.Circle):
                shape_node = etree.SubElement(g, 'circle')
                shape_node.set('r', str(shape.radius.item()))
                shape_node.set('cx', str(shape.center[0].item()))
                shape_node.set('cy', str(shape.center[1].item()))
            elif isinstance(shape, pydiffvg.Polygon):
                shape_node = etree.SubElement(g, 'polygon')
                points = shape.points.data.cpu().numpy()
                path_str = ''
                for j in range(0, shape.points.shape[0]):
                    path_str += '{} {}'.format(points[j, 0], points[j, 1])
                    if j != shape.points.shape[0] - 1:
                        path_str +=  ' '
                shape_node.set('points', path_str)
            elif isinstance(shape, pydiffvg.Path):
                shape_node = etree.SubElement(g, 'path')
                num_segments = shape.num_control_points.shape[0]
                num_control_points = shape.num_control_points.data.cpu().numpy()
                points = shape.points.data.cpu().numpy()
                num_points = shape.points.shape[0]
                path_str = 'M {} {}'.format(points[0, 0], points[0, 1])
                point_id = 1
                for j in range(0, num_segments):
                    if num_control_points[j] == 0:
                        p = point_id % num_points
                        path_str += ' L {} {}'.format(\
                                points[p, 0], points[p, 1])
                        point_id += 1
                    elif num_control_points[j] == 1:
                        p1 = (point_id + 1) % num_points
                        path_str += ' Q {} {} {} {}'.format(\
                                points[point_id, 0], points[point_id, 1],
                                points[p1, 0], points[p1, 1])
                        point_id += 2
                    elif num_control_points[j] == 2:
                        p2 = (point_id + 2) % num_points
                        path_str += ' C {} {} {} {} {} {}'.format(\
                                points[point_id, 0], points[point_id, 1],
                                points[point_id + 1, 0], points[point_id + 1, 1],
                                points[p2, 0], points[p2, 1])
                        point_id += 3
                shape_node.set('d', path_str)
            elif isinstance(shape, pydiffvg.Rect):
                shape_node = etree.SubElement(g, 'rect')
                shape_node.set('x', str(shape.p_min[0].item()))
                shape_node.set('y', str(shape.p_min[1].item()))
                shape_node.set('width', str(shape.p_max[0].item() - shape.p_min[0].item()))
                shape_node.set('height', str(shape.p_max[1].item() - shape.p_min[1].item()))
            else:
                assert(False)

            shape_node.set('stroke-width', str(2 * shape.stroke_width.data.cpu().item()))
            if shape_group.fill_color is not None:
                if isinstance(shape_group.fill_color, pydiffvg.LinearGradient):
                    shape_node.set('fill', 'url(#shape_{}_fill)'.format(i))
                else:
                    c = shape_group.fill_color.data.cpu().numpy()
                    shape_node.set('fill', 'rgb({}, {}, {})'.format(\
                        int(255 * c[0]), int(255 * c[1]), int(255 * c[2])))
                    shape_node.set('opacity', str(c[3]))
            else:
                shape_node.set('fill', 'none')
            if shape_group.stroke_color is not None:
                if isinstance(shape_group.stroke_color, pydiffvg.LinearGradient):
                    shape_node.set('stroke', 'url(#shape_{}_stroke)'.format(i))
                else:
                    c = shape_group.stroke_color.data.cpu().numpy()
                    shape_node.set('stroke', 'rgb({}, {}, {})'.format(\
                        int(255 * c[0]), int(255 * c[1]), int(255 * c[2])))
                    shape_node.set('stroke-opacity', str(c[3]))
                shape_node.set('stroke-linecap', 'round')
                shape_node.set('stroke-linejoin', 'round')
#     import pdb; pdb.set_trace()
    with open(filename, "w") as f:
        f.write(prettify(root))
def save_tensor_image(save_filename, tensor):
    image = tensor[0].permute(1, 2, 0).cpu().data.numpy()
    plt.imsave(save_filename, image)


parser = argparse.ArgumentParser(description='Generic runner for VAE models')
parser.add_argument('--config',  '-c',
                    dest="filename",
                    metavar='FILE',
                    help =  'path to the config file',
                    default='configs/vae.yaml')

args = parser.parse_args()
with open(args.filename, 'r') as file:
    try:
        config = yaml.safe_load(file)
    except yaml.YAMLError as exc:
        print(exc)

# For reproducibility
torch.manual_seed(config['logging_params']['manual_seed'])
np.random.seed(config['logging_params']['manual_seed'])
cudnn.deterministic = True
cudnn.benchmark = False

model_save_path = os.getcwd()#'{}/{}/version_{}'.format(config['logging_params']['save_dir'], config['logging_params']['name'], tt_logger.version)
parent = '/'.join(model_save_path.split('/')[:-3])
config['logging_params']['save_dir'] = os.path.join(parent, config['logging_params']['save_dir'])
config['exp_params']['data_path'] = os.path.join(parent, config['exp_params']['data_path'])
print(parent, config['exp_params']['data_path'])

model = vae_models[config['model_params']['name']](imsize=config['exp_params']['img_size'], **config['model_params'])
experiment = VAEXperiment(model,
                          config['exp_params'])

weights = [x for x in os.listdir(model_save_path) if '.ckpt' in x]
weights.sort(key=lambda x: os.path.getmtime(x))
load_weight = weights[-1]
print('loading: ', load_weight)

checkpoint = torch.load(load_weight)
experiment.load_state_dict(checkpoint['state_dict'])
_ = experiment.train_dataloader()
experiment.eval()
experiment.freeze()

version_dir = f"version_{config['logging_params']['version']}"
print(f"This is the version {version_dir}")

experiment.model.redo_features(20)

for idx, name in enumerate(experiment.sample_dataloader.dataset.samples):
    x, _ = experiment.sample_dataloader.dataset[idx]
    mu, log_var = experiment.model.encode(x[None, :, :, :])
    print(f"SVG decomposite {idx}:")
    output = decode_and_composite(experiment.model, mu, save_dir=config['logging_params']['save_dir'], name=str(idx) + "_svg", verbose=False, )

# experiment.sample_interpolate(save_dir=config['logging_params']['save_dir'], name=config['logging_params']['name'],
                            #   version=config['logging_params']['version'], save_svg=True, other_interpolations=config['logging_params']['other_interpolations'])
