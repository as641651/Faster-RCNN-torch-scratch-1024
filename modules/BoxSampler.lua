
require 'modules.BoxIoU'
local box_utils = require 'densecap.box_utils'
local utils = require 'densecap.utils'

local BoxSampler, parent = torch.class('nn.BoxSampler', 'nn.Module')


function BoxSampler:__init(options)
  parent.__init(self)
  options = options or {}
  self.fg_thresh = utils.getopt(options, 'fg_thresh', 0.5)
  self.bg_low_thresh = utils.getopt(options, 'bg_low_thresh', 0.1)
  self.bg_high_thresh = utils.getopt(options, 'bg_high_thresh', 0.5)
  self.batch_size = utils.getopt(options, 'batch_size', 128)
  self.fg_fraction = utils.getopt(options, 'fg_fraction', 0.5)

  self.nms_thresh = utils.getopt(options, 'nms_thresh', 0.7)
  --self.nms_thresh = 2
  self.num_proposals = utils.getopt(options, 'num_proposals', 2000)

  self.x_min, self.x_max = nil, nil
  self.y_min, self.y_max = nil, nil
  
  self.box_iou = nn.BoxIoU()
end

function BoxSampler:setNmsThresh(nms_thresh)
  self.nms_thresh = nms_thresh
--  self.nms_thresh = 2
end

function BoxSampler:setNumProposals(num_proposals)
  self.num_proposals = num_proposals
end

local function unpack_dims(input_boxes, target_boxes)
  local N, B1 = input_boxes:size(1), input_boxes:size(2)
  local B2 = target_boxes:size(2)
  
  assert(input_boxes:size(3) == 4 and target_boxes:size(3) == 4)
  assert(target_boxes:size(1) == N)
  
  return N, B1, B2
end


function BoxSampler:setBounds(bounds)
  self.x_min = utils.getopt(bounds, 'x_min', nil)
  self.x_max = utils.getopt(bounds, 'x_max', nil)
  self.y_min = utils.getopt(bounds, 'y_min', nil)
  self.y_max = utils.getopt(bounds, 'y_max', nil)
end


--[[
  Inputs:
  - input: list of two elements:
    - input_boxes: Tensor of shape (1, B1, 4) giving coordinates of generated
      box coordinates in (xc, yc, w, h) format
    - input_scores: Tensor of shape (1, B1, 2) giving the objectness score of
      each box.
    - target_boxes: Tensor of shape (1, B2, 4) giving coordinates of target
      box coordinates in (xc, yc, w, h) format.

  Returns: List of three elements:
    - pos_input_idx: LongTensor of shape (P,) where each element is in the
      range [1, B1] and gives indices into input_boxes for the positive boxes.
    - pos_target_idx: LongTensor of shape (P,) where each element is in the
      range [1, B2] and gives indices into target_boxes for the positive boxes.
    - neg_input_idx: LongTensor of shape (M,) where each element is in the
      range [1, B1] and gives indices into input_boxes for the negative boxes.

  Based on the ious between the generated boxes and the target boxes, we sample
  P positive boxes and M negative boxes, where P + M = batch_size. The ith
  positive box is given by input_boxes[{1, pos_input_idx[i]}], and it was matched
  to target_boxes[{1, pos_target_idx[i]}]. The ith negative box is given by
  input_boxes[{1, neg_input_idx[i]}].
--]]
function BoxSampler:updateOutput(input)
  local input_boxes = input[1]
  local input_scores = input[2]
  local target_boxes = input[3]
  local N, B1, B2 = unpack_dims(input_boxes, target_boxes)

  assert(N == 1, "Only 1-element minibatches are supported")

  -- #1: Kick out the out-of-bound boxes
  local inbounds_mask = torch.ByteTensor(N, B1):fill(1) -- N x B1

  -- Maybe find the input boxes that fall outside the boundaries
  -- and exclude them from the pos and neg masks
  if self.x_min and self.y_min and self.x_max and self.y_max then
    -- Convert from (xc, yc, w, h) to (x1, y1, x2, y2) format to make
    -- it easier to find boxes that are out of bounds
    local boxes_x1y1x2y2 = box_utils.xcycwh_to_x1y1x2y2(input_boxes)
    local x_min_mask = torch.lt(boxes_x1y1x2y2:select(3, 1), self.x_min):byte()
    local y_min_mask = torch.lt(boxes_x1y1x2y2:select(3, 2), self.y_min):byte()
    local x_max_mask = torch.gt(boxes_x1y1x2y2:select(3, 3), self.x_max):byte()
    local y_max_mask = torch.gt(boxes_x1y1x2y2:select(3, 4), self.y_max):byte()
    inbounds_mask[x_min_mask] = 0
    inbounds_mask[y_min_mask] = 0
    inbounds_mask[x_max_mask] = 0
    inbounds_mask[y_max_mask] = 0
  end

  if inbounds_mask[inbounds_mask:eq(1)]:numel() == 0 then
      inbounds_mask = torch.ByteTensor(N, B1):fill(1)
      print("no inbound rois. No masking done in boxsampler")
  end
  local inbounds_idx = inbounds_mask:view(-1):nonzero():view(-1) -- The index of in-bound boxes
  -- input_boxes changes to (N, num_inbounds, 4)
  input_boxes = input_boxes:index(2, inbounds_idx) 
  input_scores = input_scores:index(2, inbounds_idx)
  local num_inbounds = inbounds_idx:size(1)

  -- #2: Apply nms over proposals
  -- Convert rpn boxes from (xc, yc, w, h) format to (x1, y1, x2, y2)
  local nms_idx
  if self.nms_thresh < 1 then
    local rpn_boxes_x1y1x2y2 = box_utils.xcycwh_to_x1y1x2y2(input_boxes)

    -- positive - negative scores
    local scores = input_scores[{1, {}, 1}] - input_scores[{1, {}, 2}]
    
    local verbose = false
    if verbose then
      print('in LocalizationLayer forward_train, BoxSampler')
      print(string.format('Before NMS there are %d boxes', num_inbounds))
      print(string.format('Using NMS threshold %f', self.nms_thresh))
    end

    -- Run NMS and sort by objectness score
    local boxes_scores = scores.new(num_inbounds, 5)
    boxes_scores[{{}, {1, 4}}] = rpn_boxes_x1y1x2y2
    boxes_scores[{{}, 5}] = scores
    if self.num_proposals == -1 then
      nms_idx = box_utils.nms(boxes_scores, self.nms_thresh)
    else
      nms_idx = box_utils.nms(boxes_scores, self.nms_thresh, self.num_proposals)
    end
  else
    -- Skip nms
    nms_idx = torch.range(1, num_inbounds):long()
  end

  -- input_boxes changes to (N, num_after_nms, 4)
  local num_after_nms = nms_idx:size(1)
  input_boxes = input_boxes:index(2, nms_idx)

  -- #3: The following steps are almost the same as the original BoxSampler, except
  -- that the out-of-bound clip is removed because it has been done at first step.

  local ious = self.box_iou:forward({input_boxes, target_boxes}) -- N x num_after_nms x B2
  local input_max_iou, input_idx = ious:max(3)   -- N x num_after_nms x 1
  input_max_iou = input_max_iou:view(N, num_after_nms)
  input_idx = input_idx:view(N, num_after_nms)
  local _, target_idx = ious:max(2) -- N x 1 x B2
  target_idx = target_idx:view(N, B2)

  -- Pick positive and negative boxes based on IoU thresholds
  self.pos_mask = torch.gt(input_max_iou, self.fg_thresh) -- N x num_after_nms
  self.neg_mask = torch.lt(input_max_iou, self.bg_high_thresh)
    :cmul(torch.ge(input_max_iou, self.bg_low_thresh))  -- N x num_after_nms

  -- Count as positive each input box that has maximal IoU with each target box,
  -- even if it is outside the bounds or does not meet the thresholds.
  -- This is important since things will crash if we don't have at least one
  -- positive box.
  self.pos_mask:scatter(2, target_idx, 1)
  self.neg_mask:scatter(2, target_idx, 0)
  
  self.pos_mask = self.pos_mask:view(num_after_nms):byte()
  self.neg_mask = self.neg_mask:view(num_after_nms):byte()

  if self.neg_mask:sum() == 0 then
    -- There were no negatives; this can happen if all input boxes are either:
    -- (1) An input box with maximal IoU with a target box
    -- (2) Out of bounds, therefore clipped
    -- (3) max IoU to all target boxes is in the range [low_thresh, high_thresh]
    -- This should be a pretty rare case, but we still need to handle it.
    -- Ideally this should do something like sort the non-positive in-bounds boxes
    -- by their max IoU to target boxes and set the negative set to be those with
    -- minimal IoU to target boxes; however this is complicated so instead we'll
    -- just sample from non-positive boxes to get negatives.
    -- We'll also log this event in the __GLOBAL_STATS__ table; if this happens
    -- regularly then we should handle it more cleverly.

    self.neg_mask:mul(self.pos_mask, -1):add(1) -- set neg_mask to inverse of pos_mask
    local k = 'BoxSampler no negatives'
    local old_val = utils.__GLOBAL_STATS__[k] or 0
    utils.__GLOBAL_STATS__[k] = old_val + 1
    print("neg mask", self.neg_mask)
  end

  local pos_mask_nonzero = self.pos_mask:nonzero():view(-1)
  local neg_mask_nonzero = self.neg_mask:nonzero():view(-1)

  local total_pos = pos_mask_nonzero:size(1)
  local total_neg = neg_mask_nonzero:size(1)

  local num_pos = math.min(math.floor(self.batch_size * self.fg_fraction), total_pos)
  local num_neg = self.batch_size - num_pos

  -- We always sample positives without replacemet
  local pos_p = torch.ones(total_pos)
  local pos_sample_idx = torch.multinomial(pos_p, num_pos, false)

  -- We sample negatives with replacement if there are not enough negatives
  -- to fill out the minibatch
  local neg_p = torch.ones(total_neg)
  local neg_replace = (total_neg < num_neg)
  if neg_replace then
    local k = 'BoxSampler negative with replacement'
    local old_val = utils.__GLOBAL_STATS__[k] or 0
    utils.__GLOBAL_STATS__[k] = old_val + 1
  end
  local neg_sample_idx = torch.multinomial(neg_p, num_neg, neg_replace)
  
  if self.debug_pos_sample_idx then
    pos_sample_idx = self.debug_pos_sample_idx
  end
  if self.debug_neg_sample_idx then
    neg_sample_idx = self.debug_neg_sample_idx
  end

  local pos_input_idx = pos_mask_nonzero:index(1, pos_sample_idx)
  local pos_target_idx = input_idx:index(2, pos_input_idx):view(num_pos)
  local neg_input_idx = neg_mask_nonzero:index(1, neg_sample_idx)

  -- #4: To transform the index back to index before out-of-bound elimination and
  -- nms.
  pos_input_idx = inbounds_idx:index(1, nms_idx:index(1, pos_input_idx))
  neg_input_idx = inbounds_idx:index(1, nms_idx:index(1, neg_input_idx))
  
  self.output = {pos_input_idx, pos_target_idx, neg_input_idx}
  return self.output
end


function BoxSampler:updateGradInput(input, gradOutput)
  error('Not implemented')
end
