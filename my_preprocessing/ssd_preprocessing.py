# Copyright 2016 The TensorFlow Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================
"""Provides utilities to preprocess images.

The my_preprocessing steps for VGG were introduced in the following technical
report:

  Very Deep Convolutional Networks For Large-Scale Image Recognition
  Karen Simonyan and Andrew Zisserman
  arXiv technical report, 2015
  PDF: http://arxiv.org/pdf/1409.1556.pdf
  ILSVRC 2014 Slides: http://www.robots.ox.ac.uk/~karen/pdf/ILSVRC_2014.pdf
  CC-BY-4.0

More information can be obtained from the VGG website:
www.robots.ox.ac.uk/~vgg/research/very_deep/
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf
from tensorflow.python.ops import control_flow_ops
import cv2
slim = tf.contrib.slim

# _R_MEAN = 123.68
# _G_MEAN = 116.78
# _B_MEAN = 103.94

_R_MEAN = 127.5
_G_MEAN = 127.5
_B_MEAN = 127.5

def _ImageDimensions(image, rank = 3):
  """Returns the dimensions of an image tensor.

  Args:
    image: A rank-D Tensor. For 3-D  of shape: `[height, width, channels]`.
    rank: The expected rank of the image

  Returns:
    A list of corresponding to the dimensions of the
    input image.  Dimensions that are statically known are python integers,
    otherwise they are integer scalar tensors.
  """
  if image.get_shape().is_fully_defined():
    return image.get_shape().as_list()
  else:
    static_shape = image.get_shape().with_rank(rank).as_list()
    dynamic_shape = tf.unstack(tf.shape(image), rank)
    return [s if s is not None else d
            for s, d in zip(static_shape, dynamic_shape)]

def apply_with_random_selector(x, func, num_cases):
  """Computes func(x, sel), with sel sampled from [0...num_cases-1].

  Args:
    x: input Tensor.
    func: Python function to apply.
    num_cases: Python int32, number of cases to sample sel from.

  Returns:
    The result of func(x, sel), where func receives the value of the
    selector as a python integer, but sel is sampled dynamically.
  """
  sel = tf.random_uniform([], maxval=num_cases, dtype=tf.int32)
  # Pass the real x only to one of the func calls.
  return control_flow_ops.merge([
      func(control_flow_ops.switch(x, tf.equal(sel, case))[1], case)
      for case in range(num_cases)])[0]


def distort_color(image, color_ordering=0, fast_mode=True, scope=None):
  """Distort the color of a Tensor image.

  Each color distortion is non-commutative and thus ordering of the color ops
  matters. Ideally we would randomly permute the ordering of the color ops.
  Rather then adding that level of complication, we select a distinct ordering
  of color ops for each my_preprocessing thread.

  Args:
    image: 3-D Tensor containing single image in [0, 1].
    color_ordering: Python int, a type of distortion (valid values: 0-3).
    fast_mode: Avoids slower ops (random_hue and random_contrast)
    scope: Optional scope for name_scope.
  Returns:
    3-D Tensor color-distorted image on range [0, 1]
  Raises:
    ValueError: if color_ordering not in [0, 3]
  """
  with tf.name_scope(scope, 'distort_color', [image]):
    if fast_mode:
      # image=tf.Print(image,[image],message='distort image')
      if color_ordering == 0:
        image = tf.image.random_brightness(image, max_delta=32. / 255.)
        image = tf.image.random_saturation(image, lower=0.5, upper=1.5)
      else:
        image = tf.image.random_saturation(image, lower=0.5, upper=1.5)
        image = tf.image.random_brightness(image, max_delta=32. / 255.)
    else:
      if color_ordering == 0:
        image = tf.image.random_brightness(image, max_delta=32. / 255.)
        image = tf.image.random_saturation(image, lower=0.5, upper=1.5)
        image = tf.image.random_hue(image, max_delta=0.2)
        image = tf.image.random_contrast(image, lower=0.5, upper=1.5)
      elif color_ordering == 1:
        image = tf.image.random_saturation(image, lower=0.5, upper=1.5)
        image = tf.image.random_brightness(image, max_delta=32. / 255.)
        image = tf.image.random_contrast(image, lower=0.5, upper=1.5)
        image = tf.image.random_hue(image, max_delta=0.2)
      elif color_ordering == 2:
        image = tf.image.random_contrast(image, lower=0.5, upper=1.5)
        image = tf.image.random_hue(image, max_delta=0.2)
        image = tf.image.random_brightness(image, max_delta=32. / 255.)
        image = tf.image.random_saturation(image, lower=0.5, upper=1.5)
      elif color_ordering == 3:
        image = tf.image.random_hue(image, max_delta=0.2)
        image = tf.image.random_saturation(image, lower=0.5, upper=1.5)
        image = tf.image.random_contrast(image, lower=0.5, upper=1.5)
        image = tf.image.random_brightness(image, max_delta=32. / 255.)
      else:
        raise ValueError('color_ordering must be in [0, 3]')

    # The random_* ops do not necessarily clamp.
    return tf.clip_by_value(image, 0.0, 1.0)

def check_keypoints(keypoints,roi):
    p1 = tf.div(tf.cast(tf.reduce_sum(keypoints[:, 0:2], axis=1), dtype=tf.float32), 2)
    p2 = tf.div(tf.cast(tf.reduce_sum(keypoints[:, 1:3], axis=1), dtype=tf.float32), 2)
    x1 = p1[:, 0]
    y1 = p1[:, 1]
    x2 = p2[:, 0]
    y2 = p2[:, 1]
    mask_k_min = tf.logical_or(
        tf.logical_and(
            tf.logical_and(tf.greater(y1, roi[0]), tf.greater(x1, roi[1])),
            tf.logical_and(tf.greater(y2, roi[0]), tf.greater(x2, roi[1]))
        ),
        tf.logical_and(
            tf.logical_and(tf.less(y1, -1), tf.less(x1, -1)),
            tf.logical_and(tf.less(y2, -1), tf.less(x2, -1))
        ))
    mask_k_max = tf.logical_and(
        tf.logical_and(tf.less(y1, roi[2]), tf.less(x1, roi[3])),
        tf.logical_and(tf.less(y2, roi[2]), tf.less(x2, roi[3]))
    )

    mask_k = tf.logical_and(mask_k_min, mask_k_max)
    return mask_k

def check_bbox_center(bboxes,roi):
    center_x, center_y = (bboxes[:, 1] + bboxes[:, 3]) / 2, (bboxes[:, 0] + bboxes[:, 2]) / 2
    mask_min = tf.logical_and(tf.greater(center_y, roi[0]), tf.greater(center_x, roi[1]))
    mask_max = tf.logical_and(tf.less(center_y, roi[2]), tf.less(center_x, roi[3]))
    mask = tf.logical_and(mask_min, mask_max)
    return mask

def ssd_random_sample_patch(image, labels, bboxes, keypoints,ratio_list=[0.1, 0.3, 0.5, 0.7, 0.9, 1.], name=None):
  '''ssd_random_sample_patch.
  select one min_iou
  sample _width and _height from [0-width] and [0-height]
  check if the aspect ratio between 0.5-2.
  select left_top point from (width - _width, height - _height)
  check if this bbox has a min_iou with all ground_truth bboxes
  keep ground_truth those center is in this sampled patch, if none then try again
  '''
  def sample_width_height(width, height):
    with tf.name_scope('sample_width_height'):
      aspect_ratio=width*1.0/height
      sampled_width = tf.random_uniform([1], minval=0.5, maxval=0.999, dtype=tf.float32)[0] * width
      aspect_ratio_new =tf.random_uniform([1],minval=aspect_ratio-0.2,maxval=aspect_ratio+0.2,dtype=tf.float32)[0]
      sampled_height=tf.minimum(height-1,sampled_width/aspect_ratio_new)
      return tf.cast(sampled_width, tf.int32), tf.cast(sampled_height, tf.int32)

  def jaccard_with_anchors(roi, bboxes):
    with tf.name_scope('jaccard_with_anchors'):
      int_ymin = tf.maximum(roi[0], bboxes[:, 0])
      int_xmin = tf.maximum(roi[1], bboxes[:, 1])
      int_ymax = tf.minimum(roi[2], bboxes[:, 2])
      int_xmax = tf.minimum(roi[3], bboxes[:, 3])
      h = tf.maximum(int_ymax - int_ymin, 0.)
      w = tf.maximum(int_xmax - int_xmin, 0.)
      inter_vol = h * w
      union_vol = (roi[3] - roi[1]) * (roi[2] - roi[0]) + ((bboxes[:, 2] - bboxes[:, 0]) * (bboxes[:, 3] - bboxes[:, 1]) - inter_vol)
      jaccard = tf.div(inter_vol, union_vol)
      return jaccard

  def overlap_ratio_with_anchors(roi, bboxes):
    with tf.name_scope('jaccard_with_anchors'):
      int_ymin = tf.maximum(roi[0], bboxes[:, 0])
      int_xmin = tf.maximum(roi[1], bboxes[:, 1])
      int_ymax = tf.minimum(roi[2], bboxes[:, 2])
      int_xmax = tf.minimum(roi[3], bboxes[:, 3])
      h = tf.maximum(int_ymax - int_ymin, 0.)
      w = tf.maximum(int_xmax - int_xmin, 0.)
      inter_vol = h * w
      bboxes_area = (bboxes[:, 2] - bboxes[:, 0]) * (bboxes[:, 3] - bboxes[:, 1])
      overlap_ratio = tf.div(inter_vol, bboxes_area)
      return overlap_ratio

  def areas(bboxes):
    with tf.name_scope('bboxes_areas'):
      vol = (bboxes[:, 3] - bboxes[:, 1]) * (bboxes[:, 2] - bboxes[:, 0])
      return vol

  def check_roi_center(width, height, labels, bboxes, keypoints):
    with tf.name_scope('check_roi_center'):
      index = 0
      max_attempt = 20
      roi = [0., 0., 0., 0.]
      float_width = tf.cast(width, tf.float32)
      float_height = tf.cast(height, tf.float32)
      mask = tf.cast(tf.zeros_like(labels, dtype=tf.uint8), tf.bool)
      # center_x, center_y = (bboxes[:, 1] + bboxes[:, 3]) / 2, (bboxes[:, 0] + bboxes[:, 2]) / 2

      # select keypoints whoes value > 0
      # keypoint_mask = tf.reduce_max(tf.cast(tf.reduce_max(keypoints, axis=-1) >= 0, dtype=tf.int32), axis=-1) > 0
      # valid_keypoints = tf.boolean_mask(keypoints, keypoint_mask)
      # num_valid=tf.shape(valid_keypoints)[0]

      def condition(index, roi, mask):
        return tf.logical_or(tf.logical_and(tf.reduce_sum(tf.cast(mask, tf.int32)) < 1,
                                          tf.less(index, max_attempt)),
                            tf.less(index, 1))

      def body(index, roi, mask):
        sampled_width, sampled_height = sample_width_height(float_width, float_height)

        x = tf.random_uniform([], minval=0, maxval=width - sampled_width, dtype=tf.int32)
        y = tf.random_uniform([], minval=0, maxval=height - sampled_height, dtype=tf.int32)

        # y_o= tf.cast(tf.reduce_max(y2)*tf.cast(width,dtype=tf.float32),dtype=tf.int32)+1
        # y=height - sampled_height

        roi = [tf.cast(y, tf.float32) / float_height,
              tf.cast(x, tf.float32) / float_width,
              tf.cast(y + sampled_height, tf.float32) / float_height,
              tf.cast(x + sampled_width, tf.float32) / float_width]

        # mask_min = tf.logical_and(tf.greater(center_y, roi[0]), tf.greater(center_x, roi[1]))
        # mask_max = tf.logical_and(tf.less(center_y, roi[2]), tf.less(center_x, roi[3]))
        # mask = tf.logical_and(mask_min, mask_max)
        mask=check_bbox_center(bboxes,roi)
        mask=tf.Print(mask,[mask],message='mask before',summarize=100)

        mask_k=check_keypoints(keypoints,roi)

        mask=tf.logical_and(mask_k,mask)
        mask = tf.Print(mask, [mask], message='mask after', summarize=100)
        index=tf.Print(index,[index],message='index')
        return index + 1, roi, mask

      [index, roi, mask] = tf.while_loop(condition, body, [index, roi, mask], parallel_iterations=10, back_prop=False, swap_memory=True)

      mask_labels = tf.boolean_mask(labels, mask)
      mask_bboxes = tf.boolean_mask(bboxes, mask)
      mask_keypoints= tf.boolean_mask(keypoints,mask)

      return roi, mask_labels, mask_bboxes,mask_keypoints

  def check_roi_overlap(width, height, labels, bboxes, keypoints, min_iou):
    with tf.name_scope('check_roi_overlap'):
      index = 0
      max_attempt = 50
      roi = [0., 0., 1., 1.]
      mask_labels = labels
      mask_bboxes = bboxes
      mask_keypoints=keypoints

      def condition(index, roi, mask_labels, mask_bboxes,mask_keypoints):
        return tf.logical_or(tf.logical_and(tf.reduce_sum(tf.cast(overlap_ratio_with_anchors(roi, mask_bboxes) < min_iou, tf.int32)) > 0,
                                                        tf.less(index, max_attempt)),
                                          tf.less(index, 1))

      def body(index, roi, mask_labels, mask_bboxes,mask_keypoints):
        index = tf.Print(index, [index], message='check_roi_overlap index')
        roi, mask_labels, mask_bboxes, mask_keypoints = check_roi_center(width, height, labels, bboxes, keypoints)
        return index+1, roi, mask_labels, mask_bboxes,mask_keypoints

      [index, roi, mask_labels, mask_bboxes, mask_keypoints] = tf.while_loop(condition, body, [index, roi, mask_labels, mask_bboxes, mask_keypoints], parallel_iterations=16, back_prop=False, swap_memory=True)

      return tf.cond(tf.greater(tf.shape(mask_labels)[0], 0),
                  lambda : (tf.cast([roi[0] * tf.cast(height, tf.float32),
                            roi[1] * tf.cast(width, tf.float32),
                            (roi[2] - roi[0]) * tf.cast(height, tf.float32),
                            (roi[3] - roi[1]) * tf.cast(width, tf.float32)], tf.int32), mask_labels, mask_bboxes,mask_keypoints),
                  lambda : (tf.cast([0, 0, height, width], tf.int32), labels, bboxes, keypoints))


  def sample_patch(image, labels, bboxes, keypoints, min_iou):
    with tf.name_scope('sample_patch'):
      height, width, depth = _ImageDimensions(image, rank=3)

      roi_slice_range, mask_labels, mask_bboxes, mask_keypoints = check_roi_overlap(width, height, labels, bboxes, keypoints, min_iou)

      scale = tf.cast(tf.stack([height, width, height, width]), mask_bboxes.dtype)
      mask_bboxes = mask_bboxes * scale

      # Add offset.
      offset = tf.cast(tf.stack([roi_slice_range[0], roi_slice_range[1], roi_slice_range[0], roi_slice_range[1]]), mask_bboxes.dtype)
      mask_bboxes = mask_bboxes - offset

      scale2=tf.cast(tf.stack([width,height]),mask_keypoints.dtype)
      mask_keypoints_absolute=mask_keypoints*scale2
      offset2= tf.cast(tf.stack([roi_slice_range[1],roi_slice_range[0]]),mask_keypoints.dtype)
      mask_keypoints_absolute =mask_keypoints_absolute-offset2

      cliped_ymin = tf.maximum(0., mask_bboxes[:, 0])
      cliped_xmin = tf.maximum(0., mask_bboxes[:, 1])
      cliped_ymax = tf.minimum(tf.cast(roi_slice_range[2], tf.float32), mask_bboxes[:, 2])
      cliped_xmax = tf.minimum(tf.cast(roi_slice_range[3], tf.float32), mask_bboxes[:, 3])

      mask_bboxes = tf.stack([cliped_ymin, cliped_xmin, cliped_ymax, cliped_xmax], axis=-1)
      # Rescale to target dimension.
      scale = tf.cast(tf.stack([roi_slice_range[2], roi_slice_range[3],
                                roi_slice_range[2], roi_slice_range[3]]), mask_bboxes.dtype)
      scale2=tf.cast(tf.stack([roi_slice_range[3],roi_slice_range[2]]),mask_keypoints.dtype)
      mask_keypoints=tf.where(mask_keypoints>=0,mask_keypoints_absolute/scale2,mask_keypoints)
      mask_keypoints=tf.Print(mask_keypoints,[mask_keypoints],message='kepoints_after_crop',summarize=100)
      return tf.cond(tf.logical_or(tf.less(roi_slice_range[2], 1), tf.less(roi_slice_range[3], 1)),
                  lambda: (image, labels, bboxes,keypoints),
                  lambda: (tf.slice(image, [roi_slice_range[0], roi_slice_range[1], 0], [roi_slice_range[2], roi_slice_range[3], -1]),
                                  mask_labels, mask_bboxes / scale,mask_keypoints))

  with tf.name_scope('ssd_random_sample_patch'):
    image = tf.convert_to_tensor(image, name='image')

    min_iou_list = tf.convert_to_tensor(ratio_list)
    # tf.multinomial Draws samples from a multinomial distribution.
    # Example:
    # samples has shape [1, 5], where each value is either 0 or 1 with equal
    # probability.
    # samples = tf.multinomial(tf.log([[10., 10.]]), 5)
    samples_min_iou = tf.multinomial(tf.log([[1. / len(ratio_list)] * len(ratio_list)]), 1)

    sampled_min_iou = min_iou_list[tf.cast(samples_min_iou[0][0], tf.int32)]

    return tf.cond(tf.less(sampled_min_iou, 1.), lambda: sample_patch(image, labels, bboxes, keypoints,sampled_min_iou), lambda: (image, labels, bboxes,keypoints))

def ssd_random_expand(image, bboxes,keypoints, ratio=2., name=None):
  with tf.name_scope('ssd_random_expand'):
    image = tf.convert_to_tensor(image, name='image')
    if image.get_shape().ndims != 3:
      raise ValueError('\'image\' must have 3 dimensions.')

    height, width, depth = _ImageDimensions(image, rank=3)

    float_height, float_width = tf.to_float(height), tf.to_float(width)

    canvas_width, canvas_height = tf.to_int32(float_width * ratio), tf.to_int32(float_height * ratio)

    mean_color_of_image = [_R_MEAN/255., _G_MEAN/255., _B_MEAN/255.]#tf.reduce_mean(tf.reshape(image, [-1, 3]), 0)

    x = tf.random_uniform([], minval=0, maxval=canvas_width - width, dtype=tf.int32)
    y = tf.random_uniform([], minval=0, maxval=canvas_height - height, dtype=tf.int32)

    paddings = tf.convert_to_tensor([[y, canvas_height - height - y], [x, canvas_width - width - x]])

    big_canvas = tf.stack([tf.pad(image[:, :, 0], paddings, "CONSTANT", constant_values = mean_color_of_image[0]),
                          tf.pad(image[:, :, 1], paddings, "CONSTANT", constant_values = mean_color_of_image[1]),
                          tf.pad(image[:, :, 2], paddings, "CONSTANT", constant_values = mean_color_of_image[2])], axis=-1)

    scale = tf.cast(tf.stack([height, width, height, width]), bboxes.dtype)
    absolute_bboxes = bboxes * scale + tf.cast(tf.stack([y, x, y, x]), bboxes.dtype)
    scale2 =tf.cast(tf.stack([width,height]),keypoints.dtype)
    absolute_keypoints=keypoints*scale2+tf.cast(tf.stack([x,y]),keypoints.dtype)

    relative_bboxes=absolute_bboxes / tf.cast(tf.stack([canvas_height, canvas_width, canvas_height, canvas_width]), bboxes.dtype)
    relative_keypoints=absolute_keypoints/ tf.cast(tf.stack([canvas_width, canvas_height]), bboxes.dtype)

    relative_keypoints=tf.where(keypoints>0,relative_keypoints,keypoints)
    return big_canvas, relative_bboxes,relative_keypoints

# def ssd_random_sample_patch_wrapper(image, labels, bboxes):
#   with tf.name_scope('ssd_random_sample_patch_wrapper'):
#     orgi_image, orgi_labels, orgi_bboxes = image, labels, bboxes
#     def check_bboxes(bboxes):
#       areas = (bboxes[:, 3] - bboxes[:, 1]) * (bboxes[:, 2] - bboxes[:, 0])
#       return tf.logical_and(tf.logical_and(areas < 0.9, areas > 0.001),
#                             tf.logical_and((bboxes[:, 3] - bboxes[:, 1]) > 0.025, (bboxes[:, 2] - bboxes[:, 0]) > 0.025))

#     index = 0
#     max_attempt = 3
#     def condition(index, image, labels, bboxes):
#       return tf.logical_or(tf.logical_and(tf.reduce_sum(tf.cast(check_bboxes(bboxes), tf.int64)) < 1, tf.less(index, max_attempt)), tf.less(index, 1))

#     def body(index, image, labels, bboxes):
#       image, bboxes = tf.cond(tf.random_uniform([], minval=0., maxval=1., dtype=tf.float32) < 0.5,
#                       lambda: (image, bboxes),
#                       lambda: ssd_random_expand(image, bboxes, tf.random_uniform([1], minval=1.1, maxval=4., dtype=tf.float32)[0]))
#       # Distort image and bounding boxes.
#       random_sample_image, labels, bboxes = ssd_random_sample_patch(image, labels, bboxes, ratio_list=[-0.1, 0.1, 0.3, 0.5, 0.7, 0.9, 1.])
#       random_sample_image.set_shape([None, None, 3])
#       return index+1, random_sample_image, labels, bboxes

#     [index, image, labels, bboxes] = tf.while_loop(condition, body, [index, orgi_image, orgi_labels, orgi_bboxes], parallel_iterations=4, back_prop=False, swap_memory=True)

#     valid_mask = check_bboxes(bboxes)
#     labels, bboxes = tf.boolean_mask(labels, valid_mask), tf.boolean_mask(bboxes, valid_mask)
#     return tf.cond(tf.less(index, max_attempt),
#                 lambda : (image, labels, bboxes),
#                 lambda : (orgi_image, orgi_labels, orgi_bboxes))

def ssd_random_sample_patch_wrapper(image, labels, bboxes,keypoints):
  with tf.name_scope('ssd_random_sample_patch_wrapper'):
    orgi_image, orgi_labels, orgi_bboxes,orgi_keypoints = image, labels, bboxes, keypoints
    def check_bboxes(bboxes):
      areas = (bboxes[:, 3] - bboxes[:, 1]) * (bboxes[:, 2] - bboxes[:, 0])
      return tf.logical_and(tf.logical_and(areas < 0.9, areas > 0.001),
                            tf.logical_and((bboxes[:, 3] - bboxes[:, 1]) > 0.025, (bboxes[:, 2] - bboxes[:, 0]) > 0.025))

    index = 0
    max_attempt = 3
    def condition(index, image, labels, bboxes,keypoints, orgi_image, orgi_labels, orgi_bboxes, orgi_keypoints):
      return tf.logical_or(tf.logical_and(tf.reduce_sum(tf.cast(check_bboxes(bboxes), tf.int64)) < 1, tf.less(index, max_attempt)), tf.less(index, 1))

    def body(index, image, labels, bboxes, keypoints, orgi_image, orgi_labels, orgi_bboxes, orgi_keypoints):
      index=tf.Print(index,[index],message='ssd_random_sample_patch_wrapper index')
      image, bboxes,keypoints = tf.cond(tf.random_uniform([], minval=0., maxval=1., dtype=tf.float32) < 0.5,
                      lambda: (orgi_image, orgi_bboxes,orgi_keypoints),
                      lambda: ssd_random_expand(orgi_image, orgi_bboxes, orgi_keypoints,tf.random_uniform([1], minval=1.01, maxval=1.3, dtype=tf.float32)[0]))

      image, labels, bboxes, keypoints = random_rotation(image, labels,bboxes, keypoints)
      # Distort image and bounding boxes.
      # random_sample_image, labels, bboxes = ssd_random_sample_patch(image, orgi_labels, bboxes, ratio_list=[-0.1, 0.1, 0.3, 0.5, 0.7, 0.9, 1.])
      random_sample_image, labels, bboxes, keypoints = ssd_random_sample_patch(image, labels, bboxes, keypoints,
                                                                    ratio_list=[0.5, 0.7, 0.9, 1., 1., 1.])
      random_sample_image.set_shape([None, None, 3])
      return index+1, random_sample_image, labels, bboxes, keypoints, orgi_image, orgi_labels, orgi_bboxes,orgi_keypoints

    [index, image, labels, bboxes, keypoints, orgi_image, orgi_labels, orgi_bboxes, orgi_keypoints] = tf.while_loop(condition, body, [index,  image, labels, bboxes, keypoints, orgi_image, orgi_labels, orgi_bboxes,orgi_keypoints], parallel_iterations=4, back_prop=False, swap_memory=True)

    valid_mask = check_bboxes(bboxes)
    labels, bboxes = tf.boolean_mask(labels, valid_mask), tf.boolean_mask(bboxes, valid_mask)
    return tf.cond(tf.less(index, max_attempt),
                lambda : (image, labels, bboxes,keypoints),
                lambda : (orgi_image, orgi_labels, orgi_bboxes,orgi_keypoints))

def _mean_image_subtraction(image, means):
  """Subtracts the given means from each image channel.

  For example:
    means = [123.68, 116.779, 103.939]
    image = _mean_image_subtraction(image, means)

  Note that the rank of `image` must be known.

  Args:
    image: a tensor of size [height, width, C].
    means: a C-vector of values to subtract from each channel.

  Returns:
    the centered image.

  Raises:
    ValueError: If the rank of `image` is unknown, if `image` has a rank other
      than three or if the number of channels in `image` doesn't match the
      number of values in `means`.
  """
  if image.get_shape().ndims != 3:
    raise ValueError('Input must be of size [height, width, C>0]')
  num_channels = image.get_shape().as_list()[-1]
  if len(means) != num_channels:
    raise ValueError('len(means) must match the number of channels')

  channels = tf.split(axis=2, num_or_size_splits=num_channels, value=image)
  for i in range(num_channels):
    channels[i]=(channels[i])/means[i] -1
  return tf.concat(axis=2, values=channels)

def unwhiten_image(image):
  means=[_R_MEAN, _G_MEAN, _B_MEAN]
  num_channels = image.get_shape().as_list()[-1]
  channels = tf.split(axis=2, num_or_size_splits=num_channels, value=image)
  for i in range(num_channels):
    channels[i] =(channels[i]+1)*means[i]
  return tf.concat(axis=2, values=channels)

def random_flip_left_right(image, bboxes, keypoints):
  with tf.name_scope('random_flip_left_right'):
    uniform_random = tf.random_uniform([], 0, 1.0)
    mirror_cond = tf.less(uniform_random, .5)
    # Flip image.
    result = tf.cond(mirror_cond, lambda: tf.image.flip_left_right(image), lambda: image)
    # Flip bboxes.
    mirror_bboxes = tf.stack([bboxes[:, 0], 1 - bboxes[:, 3],
                              bboxes[:, 2], 1 - bboxes[:, 1]], axis=-1)
    bboxes = tf.cond(mirror_cond, lambda: mirror_bboxes, lambda: bboxes)

    mirror_keypoints =tf.stack([tf.where(keypoints[:,:,0]<0,keypoints[:,:,0],1-keypoints[:,:,0]),keypoints[:,:,1]],axis=-1)
    keypoints=tf.cond(mirror_cond, lambda: mirror_keypoints, lambda: keypoints)
    return result, bboxes, keypoints

def tf_get_rotation_matrix_2d(centerx,centery,angle):
   alpha=tf.cast(tf.cos(angle),tf.float32)
   beta=tf.cast(tf.sin(angle),tf.float32)
   c1=(1-alpha)*centerx-beta*centery
   c2=beta*centerx+(1-alpha)*centery
   return tf.squeeze(tf.stack([[alpha,beta,c1],[-beta,alpha,c2]]))

def random_rotation(image, labels, bboxes,keypoints):
  with tf.name_scope('random_rotation'):

      max_attempt=20
      shape = tf.shape(image)
      def condition(index, angle, labels, bboxes,keypoints, orgi_labels, orgi_bboxes, orgi_keypoints):
          return tf.logical_or(
              tf.logical_and(tf.shape(bboxes)[0] < 1, tf.less(index, max_attempt)),
              tf.less(index, 1))


      def body(index, angle, labels, bboxes,keypoints, orgi_labels, orgi_bboxes, orgi_keypoints):

          roi=[0.,0.,1.,1.]
          angle = tf.random_uniform([], minval=-25., maxval=25., dtype=tf.float32) * (3.1415926 / 180)

          # shape=tf.Print(shape,[shape],message='shape',summarize=100)
          rot_m = tf_get_rotation_matrix_2d(tf.cast(shape[1] / 2, tf.float32), tf.cast(shape[0] / 2, tf.float32), angle)
          # rot_m=tf.Print(rot_m,[rot_m],message='rot_m',summarize=100)
          scale = tf.cast(tf.stack([shape[1], shape[0]]), dtype=tf.float32)
          # keypoints=tf.Print(keypoints,[keypoints],message='keypoints',summarize=100)
          keypoints_absolute = orgi_keypoints * scale
          new_keypoints_x = keypoints_absolute[:, :, 0] * rot_m[0, 0] + keypoints_absolute[:, :, 1] * rot_m[0, 1] + \
                            rot_m[0, 2]
          new_keypoints_y = keypoints_absolute[:, :, 0] * rot_m[1, 0] + keypoints_absolute[:, :, 1] * rot_m[1, 1] + \
                            rot_m[1, 2]
          keypoints_new = tf.stack([new_keypoints_x, new_keypoints_y], axis=-1)
          keypoints_new = tf.div(keypoints_new, scale)
          keypoints = tf.where(orgi_keypoints > 0, keypoints_new, orgi_keypoints)
          # keypoints=tf.Print(keypoints,[keypoints],message='keypoints_new',summarize=100)
          mask_k=check_keypoints(keypoints,roi)

          xmin = orgi_bboxes[:, 1]
          ymin = orgi_bboxes[:, 0]
          xmax = orgi_bboxes[:, 3]
          ymax = orgi_bboxes[:, 2]
          lefttop = tf.stack([xmin, ymin], axis=-1)
          leftbottom = tf.stack([xmin, ymax], axis=-1)
          righttop = tf.stack([xmax, ymin], axis=-1)
          rightbottom = tf.stack([xmax, ymax], axis=-1)
          corners = tf.stack([lefttop, leftbottom, righttop, rightbottom], axis=1)
          corners_absolute = corners * scale
          new_corners_x = corners_absolute[:, :, 0] * rot_m[0, 0] + corners_absolute[:, :, 1] * rot_m[0, 1] + rot_m[
              0, 2]
          new_corners_y = corners_absolute[:, :, 0] * rot_m[1, 0] + corners_absolute[:, :, 1] * rot_m[1, 1] + rot_m[
              1, 2]
          new_corners_x = tf.div(new_corners_x, tf.cast(shape[1], dtype=tf.float32))
          new_corners_y = tf.div(new_corners_y, tf.cast(shape[0], dtype=tf.float32))
          # corners_new = tf.stack([new_corners_x, new_corners_y], axis=-1)
          # corners_new = tf.div(corners_new, scale)
          xmin_new = tf.reduce_min(new_corners_x, 1)
          xmax_new = tf.reduce_max(new_corners_x, 1)
          ymin_new = tf.reduce_min(new_corners_y, 1)
          ymax_new = tf.reduce_max(new_corners_y, 1)
          bboxes = tf.stack([ymin_new, xmin_new, ymax_new, xmax_new], axis=-1)

          mask=check_bbox_center(bboxes,roi)
          mask=tf.logical_and(mask,mask_k)
          mask=tf.Print(mask,[mask],message='rotate mask',summarize=100)

          labels=tf.boolean_mask(orgi_labels,mask)
          bboxes=tf.boolean_mask(bboxes,mask)
          keypoints=tf.boolean_mask(keypoints,mask)

          return index+1,angle,labels,bboxes,keypoints,orgi_labels, orgi_bboxes, orgi_keypoints

      def random_rotate(image, labels, bboxes, keypoints):
          index = 0
          angle=0.0
          orgi_image, orgi_labels, orgi_bboxes, orgi_keypoints = image, labels, bboxes, keypoints
          [index, angle, labels, bboxes, keypoints, orgi_labels, orgi_bboxes,
           orgi_keypoints] = tf.while_loop(
              condition, body,
              [index, angle, labels, bboxes, keypoints, orgi_labels, orgi_bboxes, orgi_keypoints],
              parallel_iterations=2, back_prop=False, swap_memory=True)
          image = tf.contrib.image.rotate([orgi_image], [angle], interpolation='BILINEAR')[0]
          return tf.cond(tf.less(index,max_attempt),
                         lambda : (image,labels,bboxes,keypoints),
                         lambda : (orgi_image, orgi_labels, orgi_bboxes, orgi_keypoints))


      uniform_random = tf.random_uniform([], 0, 1.0)
      rotate_cond = tf.less(uniform_random, 0.5)
      return tf.cond(rotate_cond,
                     lambda :random_rotate(image, labels, bboxes,keypoints),
                     lambda : (image, labels, bboxes,keypoints))


def preprocess_for_train(image, labels, bboxes, keypoints, out_shape, data_format='channels_first', scope='ssd_preprocessing_train', output_rgb=True):
  """Preprocesses the given image for training.

  Args:
    image: A `Tensor` representing an image of arbitrary size.
    labels: A `Tensor` containing all labels for all bboxes of this image.
    bboxes: A `Tensor` containing all bboxes of this image, in range [0., 1.] with shape [num_bboxes, 4].
    out_shape: The height and width of the image after my_preprocessing.
    data_format: The data_format of the desired output image.
  Returns:
    A preprocessed image.
  """
  with tf.name_scope(scope, 'ssd_preprocessing_train', [image, labels, bboxes]):
    # keypoints=tf.Print(keypoints,[tf.shape(keypoints)],message='keypoints_shape')
    # bboxes=tf.Print(bboxes,[tf.shape(bboxes)],message='bboxes_shape')
    if image.get_shape().ndims != 3:
      raise ValueError('Input must be of size [height, width, C>0]')
    # Convert to float scaled [0, 1].
    # print(labels,bboxes)
    orig_dtype = image.dtype
    if orig_dtype != tf.float32:
      image = tf.image.convert_image_dtype(image, dtype=tf.float32)

    # Randomly distort the colors. There are 4 ways to do it.
    distort_image = apply_with_random_selector(image,
                                          lambda x, ordering: distort_color(x, ordering, True),
                                          num_cases=4)



    random_sample_image, labels, bboxes,keypoints = ssd_random_sample_patch_wrapper(distort_image, labels, bboxes,keypoints)
    # image, bboxes = tf.cond(tf.random_uniform([1], minval=0., maxval=1., dtype=tf.float32)[0] < 0.25,
    #                     lambda: (image, bboxes),
    #                     lambda: ssd_random_expand(image, bboxes, tf.random_uniform([1], minval=2, maxval=4, dtype=tf.int32)[0]))

    # # Distort image and bounding boxes.
    # random_sample_image, labels, bboxes = ssd_random_sample_patch(image, labels, bboxes, ratio_list=[0.1, 0.3, 0.5, 0.7, 0.9, 1.])
    # keypoints = tf.Print(keypoints, [tf.shape(keypoints)], message='keypoints_shape after_sample')
    # Randomly flip the image horizontally.
    random_sample_flip_image, bboxes, keypoints= random_flip_left_right(random_sample_image, bboxes,keypoints)
    # keypoints = tf.Print(keypoints, [tf.shape(keypoints)], message='keypoints_shape after_flip')
    # Rescale to VGG input scale.
    random_sample_flip_resized_image = tf.image.resize_images(random_sample_flip_image, out_shape, method=tf.image.ResizeMethod.BILINEAR, align_corners=False)
    random_sample_flip_resized_image.set_shape([None, None, 3])

    final_image = tf.to_float(tf.image.convert_image_dtype(random_sample_flip_resized_image, orig_dtype, saturate=True))
    final_image = _mean_image_subtraction(final_image, [_R_MEAN, _G_MEAN, _B_MEAN])

    final_image.set_shape(out_shape + [3])
    if not output_rgb:
      image_channels = tf.unstack(final_image, axis=-1, name='split_rgb')
      final_image = tf.stack([image_channels[2], image_channels[1], image_channels[0]], axis=-1, name='merge_bgr')
    if data_format == 'channels_first':
      final_image = tf.transpose(final_image, perm=(2, 0, 1))
    return final_image, labels, bboxes,keypoints

def preprocess_for_eval(image, out_shape, data_format='channels_first', scope='ssd_preprocessing_eval', output_rgb=True):
  """Preprocesses the given image for evaluation.

  Args:
    image: A `Tensor` representing an image of arbitrary size.
    out_shape: The height and width of the image after my_preprocessing.
    data_format: The data_format of the desired output image.
  Returns:
    A preprocessed image.
  """
  with tf.name_scope(scope, 'ssd_preprocessing_eval', [image]):
    image = tf.to_float(image)
    image = tf.image.resize_images(image, out_shape, method=tf.image.ResizeMethod.BILINEAR, align_corners=False)
    image.set_shape(out_shape + [3])

    image = _mean_image_subtraction(image, [_R_MEAN, _G_MEAN, _B_MEAN])
    if not output_rgb:
      image_channels = tf.unstack(image, axis=-1, name='split_rgb')
      image = tf.stack([image_channels[2], image_channels[1], image_channels[0]], axis=-1, name='merge_bgr')
    # Image data format.
    if data_format == 'channels_first':
      image = tf.transpose(image, perm=(2, 0, 1))
    return image

def preprocess_image(image, labels, bboxes,keypoints, out_shape, is_training=False, data_format='channels_first', output_rgb=True):
  """Preprocesses the given image.

  Args:
    image: A `Tensor` representing an image of arbitrary size.
    labels: A `Tensor` containing all labels for all bboxes of this image.
    bboxes: A `Tensor` containing all bboxes of this image, in range [0., 1.] with shape [num_bboxes, 4].
    out_shape: The height and width of the image after my_preprocessing.
    is_training: Wether we are in training phase.
    data_format: The data_format of the desired output image.

  Returns:
    A preprocessed image.
  """
  if is_training:
    return preprocess_for_train(image, labels, bboxes,keypoints,out_shape, data_format=data_format, output_rgb=output_rgb)
  else:
    return preprocess_for_eval(image, out_shape, data_format=data_format, output_rgb=output_rgb)
