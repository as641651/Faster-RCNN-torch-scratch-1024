
--[[
Main entry point for training a DenseCap model
]]--
-------------------------------------------------------------------------------
-- Includes
-------------------------------------------------------------------------------
require 'torch'
require 'nngraph'
require 'image'
require 'lfs'
require 'nn'
local cjson = require 'cjson'

require 'modules.DataLoader_new'
require 'modules.ApplyBoxesTransform'
require 'modules.OurCrossEntropyCriterion'
require 'modules.BoxesRegressionCriterion'
require 'modules.InvertBoxTransform'

local utils = require 'densecap.utils'
local box_utils = require 'densecap.box_utils'
local model = require 'faster_rcnn_model'
local eval_utils = require 'eval.eval_utils'

-------------------------------------------------------------------------------
-- Initializations
-------------------------------------------------------------------------------  
local opt = model.opt
opt.data_h5 = 'data/voc07.h5'
opt.data_json = 'data/voc07.json'
opt.gpu = 0
opt.seed = 123 
opt.clip_boxes = true
opt.nms_thresh = 0.7 
opt.final_nms_thresh = 0.3
opt.max_proposals = 300

opt.train = {}
opt.train.remove_outbound_boxes = 1
opt.train.mid_objectness_weight = 1.0
opt.train.mid_box_reg_weight = 1.0
opt.train.classification_weight = 1.0
opt.train.end_box_reg_weight=1.0

print(opt)
opt.train.crits = {}
opt.train.crits.box_reg_crit = nn.BoxesRegressionCriterion(opt.train.end_box_reg_weight)
opt.train.crits.classification_crit = nn.OurCrossEntropyCriterion()
opt.train.crits.obj_crit_pos = nn.OurCrossEntropyCriterion() -- for objectness
opt.train.crits.obj_crit_neg = nn.OurCrossEntropyCriterion() -- for objectness
opt.train.crits.rpn_box_reg_crit = nn.SmoothL1Criterion() -- for RPN box regression

dtype = 'torch.FloatTensor'
torch.setdefaulttensortype(dtype)
torch.manualSeed(opt.seed)
if opt.gpu >= 0 then
  -- cuda related includes and settings
  require 'cutorch'
  require 'cunn'
  require 'cudnn'
  cutorch.manualSeed(opt.seed)
  cutorch.setDevice(opt.gpu + 1) -- note +1 because lua is 1-indexed
  dtype = 'torch.CudaTensor'
end

-- initialize the data loader class
loader = DataLoader(opt)
opt.num_classes = loader:getNumClasses()
opt.idx_to_cls = loader.info.idx_to_cls
print(opt.idx_to_cls)

-- initialize the DenseCap model object
model.cnn_1:type(dtype)
model.cnn_2:type(dtype)
model.rpn:type(dtype)
model.pooling:type(dtype)
model.recog:type(dtype)
model.sampler:type(dtype)

opt.train.crits.box_reg_crit:type(dtype)
opt.train.crits.classification_crit:type(dtype)
opt.train.crits.obj_crit_pos:type(dtype)
opt.train.crits.obj_crit_neg:type(dtype)
opt.train.crits.rpn_box_reg_crit:type(dtype)

local train = {}
function train.forward_backward(input,gt_boxes,gt_labels,fine_tune_cnn)

   local losses = {}
--   losses.obj_loss_pos = 0
--   losses.obj_loss_neg = 0
   losses.obj_loss = 0
   losses.classification_loss = 0
   losses.end_box_reg_loss = 0
-------------------------------------------------------------------------------
-- forward_
-------------------------------------------------------------------------------
--   print("input : ", input:size())
   local cnn_output_1 = model.cnn_1:forward(input)
--   print(cnn_output_1:size())
   local cnn_output = model.cnn_2:forward(cnn_output_1)
--   print("cnn_output : ", cnn_output:size())
   local rpn_out = model.rpn:forward(cnn_output)
--   print("rpn_out : ", rpn_out)

   local rpn_boxes, rpn_anchors = rpn_out[1], rpn_out[2]
   local rpn_trans, rpn_scores = rpn_out[3], rpn_out[4]

-------------------------------------------------------------------------------
-- ---------------- Sample for 256 proposals
-------------------------------------------------------------------------------
   if opt.train.remove_outbound_boxes == 1 then
     local bounds = {
        x_min=1, y_min=1,
        x_max=input:size(4),
        y_max=input:size(3)
     }
     model.sampler:setBounds(bounds)
   end

   local sampler_out = model.sampler:forward{
                          rpn_out, {gt_boxes, gt_labels}}

--   print("sampler_out : ", sampler_out)
    -- Unpack pos data
   local pos_data, pos_target_data, neg_data = unpack(sampler_out)
   local pos_boxes, pos_anchors = pos_data[1], pos_data[2]
   local pos_trans, pos_scores = pos_data[3], pos_data[4]
    -- Unpack target data
   local pos_target_boxes, pos_target_labels = unpack(pos_target_data)
    -- Unpack neg data (only scores matter)
   local neg_boxes = neg_data[1]
   local neg_scores = neg_data[4]

   local num_pos, num_neg = pos_boxes:size(1), neg_scores:size(1)
  
   local roi_boxes = torch.Tensor():type(dtype)
   roi_boxes:resize(num_pos + num_neg, 4)
   roi_boxes[{{1, num_pos}}]:copy(pos_boxes)
   roi_boxes[{{num_pos + 1, num_pos + num_neg}}]:copy(neg_boxes)
-------------------------------------------------------------------------------
-- ---------------- RPN losses
-------------------------------------------------------------------------------
   local pos_labels = torch.Tensor()
   pos_labels = pos_labels:type(dtype)
   local neg_labels = torch.Tensor()
   neg_labels = neg_labels:type(dtype)

   pos_labels:resize(num_pos):fill(2)
   neg_labels:resize(num_neg):fill(1)
   
   local scores_rpn = torch.Tensor():type(dtype)
   scores_rpn:resize(num_pos + num_neg, 2)
   scores_rpn[{{1, num_pos}}]:copy(pos_scores)
   scores_rpn[{{num_pos + 1, num_pos + num_neg}}]:copy(neg_scores)
   
   local labels_rpn = torch.Tensor():type(dtype)
   labels_rpn:resize(num_pos + num_neg)
   labels_rpn[{{1, num_pos}}]:copy(pos_labels)
   labels_rpn[{{num_pos + 1, num_pos + num_neg}}]:copy(neg_labels)
   
   local obj_loss = opt.train.crits.obj_crit_pos:forward(scores_rpn, labels_rpn)
--   local obj_loss_pos = opt.train.crits.obj_crit_pos:forward(pos_scores, pos_labels)
--   local obj_loss_neg = opt.train.crits.obj_crit_neg:forward(neg_scores, neg_labels)
   local obj_weight = opt.train.mid_objectness_weight
   losses.obj_loss = obj_weight * obj_loss
--   losses.obj_loss_pos = obj_weight * obj_loss_pos
--   losses.obj_loss_neg = obj_weight * obj_loss_neg

   local pos_trans_targets = nn.InvertBoxTransform():type(dtype):forward{
                                pos_anchors, pos_target_boxes}
   -- DIRTY DIRTY HACK: To prevent the loss from blowing up, replace boxes
   -- with huge pos_trans_targets with ground-truth
   
   local max_trans = torch.abs(pos_trans_targets):max(2)
   local max_trans_mask = torch.gt(max_trans, 10):expandAs(pos_trans_targets)
   local mask_sum = max_trans_mask:sum() / 4
   if mask_sum > 0 then
     local msg = 'WARNING: Masking out %d boxes in LocalizationLayer'
     print(string.format(msg, mask_sum))
     pos_trans[max_trans_mask] = 0
     pos_trans_targets[max_trans_mask] = 0
   end

   -- Compute RPN box regression loss
   local weight = opt.train.mid_box_reg_weight
   local loss = weight * opt.train.crits.rpn_box_reg_crit:forward(pos_trans, pos_trans_targets)
   losses.box_reg_loss = loss

-------------------------------------------------------------------------------
-- ---------------- Roi Pooling and FC net
-------------------------------------------------------------------------------
   local roi_features = model.pooling:forward{cnn_output[1], roi_boxes}
--   print("roi_feats : ", roi_features:size())
   local net_out = model.recog:forward(roi_features)
--   print("net_out : ", net_out)

   net_out[2] = net_out[2]:view(net_out[2]:size(1),opt.num_classes,4)
-------------------------------------------------------------------------------
-- ---------------- Final losses
-------------------------------------------------------------------------------
   local num_out = net_out[1]:size(1)
   local target = gt_labels.new(num_out):fill(1) --  1 means background
   target[{{1, num_pos}}]:copy(pos_target_labels)

   losses.classification_loss = opt.train.crits.classification_crit:forward(net_out[1], target)
   losses.classification_loss = losses.classification_loss*opt.train.classification_weight

   --weight multiplied inside
   losses.end_box_reg_loss = opt.train.crits.box_reg_crit:forward(
                                {roi_boxes[{{1,num_pos}}], net_out[2][{{1,num_pos}}], pos_target_labels},
                                pos_target_boxes)
-------------------------------------------------------------------------------
-- backward
-------------------------------------------------------------------------------

-------------------------------------------------------------------------------
-- ---------------- grad scores and final boxes (net_out)
-------------------------------------------------------------------------------
   local grad_net_out = {}
   local din = opt.train.crits.box_reg_crit:backward(
                         {roi_boxes[{{1,num_pos}}], net_out[2][{{1,num_pos}}], pos_target_labels},
                         pos_target_boxes)
   local grad_pos_roi_boxes, grad_final_pos_box_trans, _ = unpack(din)
   grad_pos_roi_boxes:zero() -- remove this for bilinear pooling
--   grad_final_pos_box_trans:zero() --debug
   grad_net_out[2] = net_out[2].new(#net_out[2]):zero()
   grad_net_out[2][{{1,num_pos}}]:copy(grad_final_pos_box_trans) 
   grad_net_out[2] = grad_net_out[2]:view(grad_net_out[2]:size(1),opt.num_classes*4)
  
   local grad_class_scores = opt.train.crits.classification_crit:backward(net_out[1], target)
--   grad_class_scores:zero() --debug
   grad_class_scores:mul(opt.train.classification_weight)
   grad_net_out[1] = grad_class_scores

-------------------------------------------------------------------------------
-- ---------------- grad roi feats
-------------------------------------------------------------------------------
   grad_roi_features = model.recog:backward(roi_features,grad_net_out)
-------------------------------------------------------------------------------
-- ---------------- grad cnn output
-------------------------------------------------------------------------------
   local grad_cnn_output = cnn_output.new(#cnn_output):zero()

-------------------------------------------------------------------------------
-- ---------------------------- grad cnn output from ROI pooling
-------------------------------------------------------------------------------
   local grad_pool = model.pooling:backward(
                    {cnn_output[1], roi_boxes},
                    grad_roi_features)
   --grad_roi_boxes:add(din[2])

   grad_cnn_output:add(grad_pool[1]:viewAs(cnn_output))

-------------------------------------------------------------------------------
-- ---------------------------- grad cnn output from RPN
-------------------------------------------------------------------------------

-------------------------------------------------------------------------------
-- ----------------------------------------- grad pos+neg scores and grad pos trans
-------------------------------------------------------------------------------
   local grad_rpn_scores = opt.train.crits.obj_crit_pos:backward(scores_rpn, labels_rpn)
--   grad_rpn_scores:zero() --debug
--   local grad_pos_scores = opt.train.crits.obj_crit_pos:backward(pos_scores, pos_labels)
--   grad_pos_scores:zero() --debug
--   local grad_neg_scores = opt.train.crits.obj_crit_neg:backward(neg_scores, neg_labels)
--   grad_neg_scores:zero() --debug
   grad_rpn_scores:mul(opt.train.mid_objectness_weight)
--   grad_pos_scores:mul(opt.train.mid_objectness_weight)
--   grad_neg_scores:mul(opt.train.mid_objectness_weight)

   local grad_pos_trans =  opt.train.crits.rpn_box_reg_crit:backward(pos_trans, pos_trans_targets)
   grad_pos_trans:mul(opt.train.mid_box_reg_weight)
--   grad_pos_trans:zero() --debug
   local grad_neg_roi_boxes = neg_boxes.new(#neg_boxes):zero()

-------------------------------------------------------------------------------
-- ----------------------------------------- grad rpn out
-------------------------------------------------------------------------------
   local grad_pos_data, grad_neg_data = {}, {}
   grad_pos_data[1] = grad_pos_roi_boxes
   grad_pos_data[3] = grad_pos_trans
   grad_pos_data[4] = grad_rpn_scores[{{1,num_pos}}]
  -- grad_pos_data[4] = grad_pos_scores
   grad_neg_data[1] = grad_neg_roi_boxes
   grad_neg_data[4] = grad_rpn_scores[{{num_pos+1,num_pos+num_neg}}]
   --grad_neg_data[4] = grad_neg_scores

   local grad_rpn_out = model.sampler:backward(
                              {rpn_out, {gt_boxes, gt_labels}},
                              {grad_pos_data, grad_neg_data})

   --print(grad_rpn_out[4])
   --print(grad_neg_scores)

   local grad_rpn = model.rpn:backward(cnn_output,grad_rpn_out) 
--   print(grad_rpn[grad_rpn:gt(0)])
   grad_cnn_output:add(grad_rpn)

-------------------------------------------------------------------------------
-- ---------------- grad input
-------------------------------------------------------------------------------
   if fine_tune_cnn then
     local grad_cnn_output_1 = model.cnn_2:backward(cnn_output_1,grad_cnn_output)
   end
--   local grad_input = model.cnn_1:backward(input,grad_cnn_output_1)

   local total = 0
   for k,v in pairs(losses) do
     if k ~= 'total_loss' then
       total = total + v 
     end
   end
   losses.total_loss = total

   return losses

end


local deploy = {}
deploy.opt = opt
-------------------------------------------------------------------------------
-- forward_test
-------------------------------------------------------------------------------
function deploy.forward_test(input)  
   local cnn_output_1 = model.cnn_1:forward(input)
   local cnn_output = model.cnn_2:forward(cnn_output_1)
   local rpn_out = model.rpn:forward(cnn_output)
   
   local rpn_boxes, rpn_anchors = rpn_out[1], rpn_out[2]
   local rpn_trans, rpn_scores = rpn_out[3], rpn_out[4]
   local num_boxes = rpn_boxes:size(2)
   
   if opt.clip_boxes then
    local bounds = {
       x_min=1, y_min=1,
       x_max=input:size(4),
       y_max=input:size(3)
    }
    rpn_boxes, valid = box_utils.clip_boxes(rpn_boxes, bounds, 'xcycwh')

    -- Clamp parallel arrays only to valid boxes (not oob of the image)
    local function clamp_data(data)
      -- data should be 1 x kHW x D
      -- valid is byte of shape kHW
      assert(data:size(1) == 1, 'must have 1 image per batch')
      assert(data:dim() == 3)
      local mask = valid:view(1, -1, 1):expandAs(data)
      return data[mask]:view(1, -1, data:size(3))
    end

    rpn_boxes = clamp_data(rpn_boxes)
    rpn_anchors = clamp_data(rpn_anchors)
    rpn_trans = clamp_data(rpn_trans)
    rpn_scores = clamp_data(rpn_scores)

    num_boxes = rpn_boxes:size(2)
   end
  
-- Convert rpn boxes from (xc, yc, w, h) format to (x1, y1, x2, y2)
  local rpn_boxes_x1y1x2y2 = box_utils.xcycwh_to_x1y1x2y2(rpn_boxes)

  -- Convert objectness positive / negative scores to probabilities
  local rpn_scores_exp = torch.exp(rpn_scores)
  local pos_exp = rpn_scores_exp[{1, {}, 1}]
  local neg_exp = rpn_scores_exp[{1, {}, 2}]
  local scores = (pos_exp + neg_exp):pow(-1):cmul(pos_exp)
  
  local verbose = true
  if verbose then
    print('in LocalizationLayer forward_test')
    print(string.format('Before NMS there are %d boxes', num_boxes))
    print(string.format('Using NMS threshold %f', opt.nms_thresh))
  end


  -- Run NMS and sort by objectness score
  local boxes_scores = scores.new(num_boxes, 5)
  boxes_scores[{{}, {1, 4}}] = rpn_boxes_x1y1x2y2
  boxes_scores[{{}, 5}] = scores
  local idx
  if opt.max_proposals == -1 then
    idx = box_utils.nms(boxes_scores, opt.nms_thresh)
  else
    idx = box_utils.nms(boxes_scores, opt.nms_thresh, opt.max_proposals)
  end

  -- Use NMS indices to pull out corresponding data from RPN
  -- All these are being converted from (1, B2, D) to (B3, D)
  -- where B2 are the number of boxes after boundary clipping and B3
  -- is the number of boxes after NMS
  local rpn_boxes_nms = rpn_boxes:index(2, idx)[1]
  local rpn_anchors_nms = rpn_anchors:index(2, idx)[1]
  local rpn_trans_nms = rpn_trans:index(2, idx)[1]
  -- local rpn_scores_nms = rpn_scores:index(2, idx)[1]
  local rpn_scores_nms = scores:index(1, idx)
  local scores_nms = scores:index(1, idx)

  if verbose then
    print(string.format('After NMS there are %d boxes', rpn_boxes_nms:size(1)))
  end

  -- self.nets.roi_pooling:setImageSize(self.image_height, self.image_width)
  local roi_features = model.pooling:forward{cnn_output[1], rpn_boxes_nms}
  local net_out = model.recog:forward(roi_features)
 
  net_out[2] = net_out[2]:view(net_out[2]:size(1),opt.num_classes,4)
  local boxesTrans = nn.Sequential()
  boxesTrans:add(nn.ApplyBoxesTransform():type(dtype))
  local final_boxes = boxesTrans:forward({rpn_boxes_nms, net_out[2]})

  local final_boxes_float = final_boxes:float()
  local class_scores_float = net_out[1]:float()
  class_scores_float = nn.SoftMax():type(class_scores_float:type()):forward(class_scores_float)
    
  local rpn_boxes_float = rpn_boxes_nms:float()
  local rpn_scores_float = rpn_scores_nms:float()

  local final_boxes_output = {rpn_boxes_float}
  local class_scores_output = {rpn_scores_float}
   
  local after_nms_boxes = 0 
  for cls = 2, opt.num_classes do 
      local boxes_scores = torch.FloatTensor(final_boxes_float:size(1), 5)
      local boxes_x1y1x2y2 = box_utils.xcycwh_to_x1y1x2y2(final_boxes_float:select(2, cls):contiguous())
      boxes_scores[{{}, {1, 4}}]:copy(boxes_x1y1x2y2)
      boxes_scores[{{}, 5}]:copy(class_scores_float[{{}, cls}])
      local idx = box_utils.nms(boxes_scores, opt.final_nms_thresh)
      table.insert(final_boxes_output, final_boxes_float:index(1, idx):select(2, cls):typeAs(final_boxes))
      table.insert(class_scores_output, class_scores_float:index(1, idx):select(2, cls):typeAs(net_out[1]))
      after_nms_boxes = after_nms_boxes + final_boxes_output[cls]:size(1)      
  end
  if verbose then
    print(string.format('After FINAL NMS there are %d boxes', after_nms_boxes))
  end

  return final_boxes_output, class_scores_output
end

classifier = {}
classifier.train = train
classifier.deploy = deploy
classifier.loader = loader
classifier.dtype = dtype
classifier.model = model
classifier.opt = opt

return classifier

