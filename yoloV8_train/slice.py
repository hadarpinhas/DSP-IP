import numpy as np

def slice_img_with_overlap(imgshape, num_slices=3, pixel_offset=60):
    slices = list()
    h, w, c = imgshape
    if num_slices == 1:
        slices.append([0,0,w,h])
        return slices
    h_step, w_step = h//num_slices, w//num_slices

    for si in range(num_slices):
        for sj in range(num_slices):
            start_w = si * w_step
            end_w = (si+1) * w_step
            start_h = sj * h_step
            end_h = (sj+1) * h_step
            start_w = max(0,start_w - pixel_offset)
            start_h = max(0,start_h - pixel_offset)

            slices.append([int(start_w), int(start_h), int(end_w), int(end_h)])

    return slices

class edge:
    def __init__(self,i0,i1,j0,j1):
        self.i0 = i0
        self.i1 = i1
        self.j0 = j0
        self.j1 = j1
        if i0 == i1 :
            self.type = 'LR'
        if j0 == j1 :
            self.type = 'TB'

        self.boxes = []

    def __repr__(self):
        return f'{self.type}: {self.i0} {self.i1} {self.j0} {self.j1}'


def slice_edges(im_final_size ,crop_coords):
    edges = []
    width,height = im_final_size
    left,top,right,bottom = crop_coords
    if right < width:
        edges.append(edge(right,right,top,bottom))
    if bottom < height:
        edges.append(edge(left,right,bottom,bottom))
    return edges

def get_crop_size(imsize:np.array ,slice_count_xy):
    return imsize//slice_count_xy


def read_create_edges(imsize:np.array,slice_count_xy):
    crop_size = get_crop_size(imsize,slice_count_xy)
    img_final_size = crop_size * slice_count_xy
    current_crop = [0,0,0,0]
    edges = []

    for i in range(slice_count_xy):
        for j in range(slice_count_xy):
            edges.extend(slice_edges(img_final_size,[crop_size[0]*j,crop_size[1]*i, crop_size[0] * (j+1),crop_size[1] *(i+1)]))

    return edges

def between(x,mn,mx):
    return mn<x<mx

def get_intersecting_boxes(edge,bboxes):
    intersecting_boxes = []
    e = edge
    eps = 5
    for b in bboxes:
        if e.type == 'LR' and (between(b[0],e.i0-eps,e.i0+eps) or between(b[2],e.i0-eps,e.i0+eps)) and (b[1] <= e.j0 and b[3] <= e.j1):
            intersecting_boxes.append(b)
        elif e.type == 'TB' and (between(b[1],e.j0-eps,e.j0+eps) or between(b[3],e.j0-eps,e.j0+eps)) and (b[0] <= e.i0 and b[2] <= e.i1):
            intersecting_boxes.append(b)

    return intersecting_boxes

def compare_bboxes_with_edges(edges,bboxes):
    for e in edges :
        iboxes = get_intersecting_boxes(e,bboxes)
        if len(iboxes) > 0:
            e.boxes.extend(iboxes)

read_create_edges(np.array([4032,3024]),4)

