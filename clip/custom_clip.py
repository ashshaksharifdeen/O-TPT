
import math
from typing import List, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from clip import load, tokenize
from .simple_tokenizer import SimpleTokenizer as _Tokenizer
from data.imagnet_prompts import imagenet_classes
from data.fewshot_datasets import fewshot_datasets
from data.cls_to_names import *

import ipdb

_tokenizer = _Tokenizer()

DOWNLOAD_ROOT='~/.cache/clip'

class ClipImageEncoder(nn.Module):
    def __init__(self, device, arch="ViT-L/14", image_resolution=224, n_class=1000):
        super(ClipImageEncoder, self).__init__()
        clip, embed_dim, _ = load(arch, device=device, download_root=DOWNLOAD_ROOT)
        self.encoder = clip.visual
        del clip.transformer
        torch.cuda.empty_cache()
        
        self.cls_head = nn.Linear(embed_dim, n_class)
    
    @property
    def dtype(self):
        return self.encoder.conv1.weight.dtype

    def forward(self, image):
        x = self.encoder(image.type(self.dtype))
        output = self.cls_head(x)
        return output


class TextEncoder(nn.Module):
    def __init__(self, clip_model):
        super().__init__()
        self.transformer = clip_model.transformer
        self.positional_embedding = clip_model.positional_embedding
        self.ln_final = clip_model.ln_final
        self.text_projection = clip_model.text_projection
        self.dtype = clip_model.dtype

    def forward(self, prompts, tokenized_prompts):
        x = prompts + self.positional_embedding.type(self.dtype)
        x = x.permute(1, 0, 2)  # NLD -> LND
        x = self.transformer(x)
        x = x.permute(1, 0, 2)  # LND -> NLD
        x = self.ln_final(x).type(self.dtype)

        # x.shape = [batch_size, n_ctx, transformer.width]
        # take features from the eot embedding (eot_token is the highest number in each sequence)
        x = x[torch.arange(x.shape[0]), tokenized_prompts.argmax(dim=-1)] @ self.text_projection

        return x


class PromptLearner(nn.Module):
    def __init__(self, clip_model, classnames, batch_size=None, n_ctx=16, ctx_init=None, ctx_position='end', learned_cls=False):
        super().__init__()
        n_cls = len(classnames)
        self.learned_cls = learned_cls
        dtype = clip_model.dtype
        self.dtype = dtype
        self.device = clip_model.visual.conv1.weight.device
        ctx_dim = clip_model.ln_final.weight.shape[0]
        self.ctx_dim = ctx_dim
        self.batch_size = batch_size

        # self.ctx, prompt_prefix = self.reset_prompt(ctx_dim, ctx_init, clip_model)

        if ctx_init:
            # use given words to initialize context vectors
            print("Initializing the contect with given words: [{}]".format(ctx_init))
            ctx_init = ctx_init.replace("_", " ")
            if '[CLS]' in ctx_init:
                ctx_list = ctx_init.split(" ")
                split_idx = ctx_list.index("[CLS]")
                ctx_init = ctx_init.replace("[CLS] ", "")
                ctx_position = "middle"
            else:
                split_idx = None
            self.split_idx = split_idx
            n_ctx = len(ctx_init.split(" "))
            prompt = tokenize(ctx_init).to(self.device)
            with torch.no_grad():
                embedding = clip_model.token_embedding(prompt).type(dtype)
            ctx_vectors = embedding[0, 1 : 1 + n_ctx, :]
            prompt_prefix = ctx_init
        else:
            print("Random initialization: initializing a generic context")
            ctx_vectors = torch.empty(n_ctx, ctx_dim, dtype=dtype)
            nn.init.normal_(ctx_vectors, std=0.02)
            prompt_prefix = " ".join(["X"] * n_ctx)
        
        self.prompt_prefix = prompt_prefix
        
        print(f'Initial context: "{prompt_prefix}"')
        print(f"Number of context words (tokens): {n_ctx}")

        # batch-wise prompt tuning for test-time adaptation
        if self.batch_size is not None: 
            ctx_vectors = ctx_vectors.repeat(batch_size, 1, 1)  #(N, L, D)
        self.ctx_init_state = ctx_vectors.detach().clone()
        self.ctx = nn.Parameter(ctx_vectors) # to be optimized

        if not self.learned_cls:
            classnames = [name.replace("_", " ") for name in classnames]
            name_lens = [len(_tokenizer.encode(name)) for name in classnames]
            prompts = [prompt_prefix + " " + name + "." for name in classnames]
        else:
            print("Random initialization: initializing a learnable class token")
            cls_vectors = torch.empty(n_cls, 1, ctx_dim, dtype=dtype) # assume each learnable cls_token is only 1 word
            nn.init.normal_(cls_vectors, std=0.02)
            cls_token = "X"
            name_lens = [1 for _ in classnames]
            prompts = [prompt_prefix + " " + cls_token + "." for _ in classnames]

            self.cls_init_state = cls_vectors.detach().clone()
            self.cls = nn.Parameter(cls_vectors) # to be optimized

        tokenized_prompts = torch.cat([tokenize(p) for p in prompts]).to(self.device)
        with torch.no_grad():
            embedding = clip_model.token_embedding(tokenized_prompts).type(dtype)

        # These token vectors will be saved when in save_model(),
        # but they should be ignored in load_model() as we want to use
        # those computed using the current class names
        self.register_buffer("token_prefix", embedding[:, :1, :])  # SOS
        if self.learned_cls:
            self.register_buffer("token_suffix", embedding[:, 1 + n_ctx + 1:, :])  # ..., EOS
        else:
            self.register_buffer("token_suffix", embedding[:, 1 + n_ctx :, :])  # CLS, EOS

        self.ctx_init = ctx_init
        self.tokenized_prompts = tokenized_prompts  # torch.Tensor
        self.name_lens = name_lens
        self.class_token_position = ctx_position
        self.n_cls = n_cls
        self.n_ctx = n_ctx
        self.classnames = classnames

    def reset(self):
        ctx_vectors = self.ctx_init_state
        self.ctx.copy_(ctx_vectors) # to be optimized
        if self.learned_cls:
            cls_vectors = self.cls_init_state
            self.cls.copy_(cls_vectors)

    def reset_classnames(self, classnames, arch):
        self.n_cls = len(classnames)
        if not self.learned_cls:
            classnames = [name.replace("_", " ") for name in classnames]
            name_lens = [len(_tokenizer.encode(name)) for name in classnames]
            prompts = [self.prompt_prefix + " " + name + "." for name in classnames]
        else:
            cls_vectors = torch.empty(self.n_cls, 1, self.ctx_dim, dtype=self.dtype) # assume each learnable cls_token is only 1 word
            nn.init.normal_(cls_vectors, std=0.02)
            cls_token = "X"
            name_lens = [1 for _ in classnames]
            prompts = [self.prompt_prefix + " " + cls_token + "." for _ in classnames]
            # TODO: re-init the cls parameters
            # self.cls = nn.Parameter(cls_vectors) # to be optimized
            self.cls_init_state = cls_vectors.detach().clone()
        tokenized_prompts = torch.cat([tokenize(p) for p in prompts]).to(self.device)

        clip, _, _ = load(arch, device=self.device, download_root=DOWNLOAD_ROOT)

        with torch.no_grad():
            embedding = clip.token_embedding(tokenized_prompts).type(self.dtype)

        self.token_prefix = embedding[:, :1, :]
        self.token_suffix = embedding[:, 1 + self.n_ctx :, :]  # CLS, EOS

        self.name_lens = name_lens
        self.tokenized_prompts = tokenized_prompts
        self.classnames = classnames

    def forward(self, init=None):
        # the init will be used when computing CLIP directional loss
        if init is not None:
            ctx = init
        else:
            ctx = self.ctx
        if ctx.dim() == 2:
            ctx = ctx.unsqueeze(0).expand(self.n_cls, -1, -1)
        elif not ctx.size()[0] == self.n_cls:
            ctx = ctx.unsqueeze(1).expand(-1, self.n_cls, -1, -1)

        prefix = self.token_prefix
        suffix = self.token_suffix
        if self.batch_size is not None: 
            # This way only works for single-gpu setting (could pass batch size as an argument for forward())
            prefix = prefix.repeat(self.batch_size, 1, 1, 1)
            suffix = suffix.repeat(self.batch_size, 1, 1, 1)

        if self.learned_cls:
            assert self.class_token_position == "end"
        if self.class_token_position == "end":
            if self.learned_cls:
                cls = self.cls
                prompts = torch.cat(
                    [
                        prefix,  # (n_cls, 1, dim)
                        ctx,     # (n_cls, n_ctx, dim)
                        cls,     # (n_cls, 1, dim)
                        suffix,  # (n_cls, *, dim)
                    ],
                    dim=-2,
                )
            else:
                prompts = torch.cat(
                    [
                        prefix,  # (n_cls, 1, dim)
                        ctx,     # (n_cls, n_ctx, dim)
                        suffix,  # (n_cls, *, dim)
                    ],
                    dim=-2,
                )
        elif self.class_token_position == "middle":
            # TODO: to work with a batch of prompts
            if self.split_idx is not None:
                half_n_ctx = self.split_idx # split the ctx at the position of [CLS] in `ctx_init`
            else:
                half_n_ctx = self.n_ctx // 2
            prompts = []
            for i in range(self.n_cls):
                name_len = self.name_lens[i]
                prefix_i = prefix[i : i + 1, :, :]
                class_i = suffix[i : i + 1, :name_len, :]
                suffix_i = suffix[i : i + 1, name_len:, :]
                ctx_i_half1 = ctx[i : i + 1, :half_n_ctx, :]
                ctx_i_half2 = ctx[i : i + 1, half_n_ctx:, :]
                prompt = torch.cat(
                    [
                        prefix_i,     # (1, 1, dim)
                        ctx_i_half1,  # (1, n_ctx//2, dim)
                        class_i,      # (1, name_len, dim)
                        ctx_i_half2,  # (1, n_ctx//2, dim)
                        suffix_i,     # (1, *, dim)
                    ],
                    dim=1,
                )
                prompts.append(prompt)
            prompts = torch.cat(prompts, dim=0)

        elif self.class_token_position == "front":
            prompts = []
            for i in range(self.n_cls):
                name_len = self.name_lens[i]
                prefix_i = prefix[i : i + 1, :, :]
                class_i = suffix[i : i + 1, :name_len, :]
                suffix_i = suffix[i : i + 1, name_len:, :]
                ctx_i = ctx[i : i + 1, :, :]
                prompt = torch.cat(
                    [
                        prefix_i,  # (1, 1, dim)
                        class_i,   # (1, name_len, dim)
                        ctx_i,     # (1, n_ctx, dim)
                        suffix_i,  # (1, *, dim)
                    ],
                    dim=1,
                )
                prompts.append(prompt)
            prompts = torch.cat(prompts, dim=0)

        else:
            raise ValueError

        return prompts


class ClipTestTimeTuning(nn.Module):
    def __init__(self, device, classnames, constant, batch_size, criterion='cosine', arch="ViT-L/14",
                        n_ctx=16, ctx_init=None, ctx_position='end', learned_cls=False):
        super(ClipTestTimeTuning, self).__init__()
        clip, _, _ = load(arch, device=device, download_root=DOWNLOAD_ROOT)
        self.image_encoder = clip.visual
        self.text_encoder = TextEncoder(clip)
        self.logit_scale = clip.logit_scale.data
        # prompt tuning
        self.prompt_learner = PromptLearner(clip, classnames, batch_size, n_ctx, ctx_init, ctx_position, learned_cls)
        self.criterion = criterion
        self.constant_list = constant
        
    @property
    def dtype(self):
        return self.image_encoder.conv1.weight.dtype

    # restore the initial state of the prompt_learner (tunable prompt)
    def reset(self):
        self.prompt_learner.reset()

    def reset_classnames(self, classnames, arch):
        self.prompt_learner.reset_classnames(classnames, arch)

    def get_text_features(self):
        text_features = []
        prompts = self.prompt_learner()
        tokenized_prompts = self.prompt_learner.tokenized_prompts
        t_features = self.text_encoder(prompts, tokenized_prompts)
        text_features.append(t_features / t_features.norm(dim=-1, keepdim=True))
        text_features = torch.stack(text_features, dim=0)

        return torch.mean(text_features, dim=0)
    
    def select_feature(self,logits,selection_p):
        #computing the softmax and log-values +summing up
        batch_entropy = -(logits.softmax(1) * logits.log_softmax(1)).sum(1) #batch entropy shape [64]
        idx = torch.argsort(batch_entropy, descending=False)[:int(batch_entropy.size()[0] * selection_p)]
        #remaining indexes
        rem_idx = torch.argsort(batch_entropy, descending=False)[int(batch_entropy.size()[0] * selection_p):]
        #this returns the maximum confidence for the each augmented image and it's index
        return idx,rem_idx

    def inference(self, image,cons,args):
        with torch.no_grad():
            image_features = self.image_encoder(image.type(self.dtype))

        text_features = self.get_text_features()
        #it test through each image
        image_features = image_features / image_features.norm(dim=-1, keepdim=True)
        
        #[c-tpt] --------------------------------------------
        if self.l2_norm_cal:

            #calculating selected ifeature disperssion
            softmax = torch.nn.Softmax(dim=1)
            logit_scale_ = self.logit_scale.exp()
            logits_slt = logit_scale_ * image_features @ text_features.t()

            if logits_slt.shape[0]==1:

                cort_sftmax = softmax(logits_slt)
                #max score
                max_confidence, max_index = torch.max(logits_slt, 1)
                #print("maximum index shape:",max_index.shape)
                #print("maximum index:",max_index)
                #min score
                max_confidence, minimum_index = torch.min(logits_slt, 1)
                #print("minimum index shape:",minimum_index.shape)
                #print("minimum index:",minimum_index)
                #print("1D image feature:",image_features.shape)
                correct_img_features = image_features
                in_img_features = image_features    

            else:

                #print(" first logit shape:",logits_slt.shape)
                select_idx,rem_indx = self.select_feature(logits_slt,args.selection_p)
                #selecting top correct logit
                correct_logit = logits_slt[select_idx]
                #print("correct selected logit shape:",correct_logit.shape)
                #print("correct selected :",correct_logit)
                #selecting the incorrect logit
                incorrect_logit = logits_slt[rem_indx]
                #print("incorrect selected logit shape:",incorrect_logit.shape)
                #print("incorrect selected :",incorrect_logit)
                #find softmax of it
                cort_sftmax = softmax(correct_logit)
                incort_sftmax = softmax(incorrect_logit)


                #correct index prediction
                max_confidence, crt_max_index = torch.max(cort_sftmax, 1)
                #print("correct index logit shape:",crt_max_index.shape) 

                
                    
                #print("incorrect index logit:",crt_max_index)
                #Count the occurrences of each value
                unique_values, counts = crt_max_index.unique(return_counts=True)
                # Find the value with the maximum count
                max_count_index = counts.argmax()  # Index of the maximum count in the counts tensor
                max_value = unique_values[max_count_index]  # Value with the maximum count
                max_index = max_value.item()

                #incorrect index prediction
                fmax_confidence, incrt_max_index = torch.min(incort_sftmax, 1)
                #print("correct index logit shape:",incrt_max_index.shape) 
                #print("incorrect index logit:",incrt_max_index)

                
                    
                #Count the occurrences of each value
                inunique_values, incounts = incrt_max_index.unique(return_counts=True)
                # Find the value with the maximum count
                inmax_count_index = incounts.argmax()  # Index of the maximum count in the counts tensor
                inmax_value = inunique_values[inmax_count_index]  # Value with the maximum count
                minimum_index = inmax_value.item()
            
            
                #select correct and in correct image features
                correct_img_features = image_features[select_idx] #[6,512]
                in_img_features = image_features[rem_indx]   
                #print("correct image shape:",correct_img_features.shape) 
                #print("incorrect image shape:",in_img_features.shape)
            
                #predicting softmax output
            
                """softmax_output = softmax(logits_slt)
                #max score
                max_confidence, max_index = torch.max(softmax_output, 1)
                print("maximum index shape:",max_index.shape)
                print("maximum index:",max_index)
                #min score
                max_confidence, min_index = torch.min(softmax_output, 1)
                print("minimum index shape:",min_index.shape)
                print("minimum index:",min_index)"""

            #text features selcect
            crt_txt_features = text_features[max_index]
            icrt_txt_features = text_features[ minimum_index ]
            #print("correct text feature shape:",crt_txt_features.shape)
            #print("correct text feature shape:",icrt_txt_features.shape)

            #distance
            crt_im_txt_distance =   correct_img_features - crt_txt_features
            icrt_im_txt_distance =  in_img_features - icrt_txt_features 
            #print("correct image text distance shape:",crt_im_txt_distance.shape)                       
            #print("incorrect image text distance shape:",icrt_im_txt_distance.shape)

            #l2 norm
            l2_norm_correct = torch.linalg.norm(crt_im_txt_distance, dim=-1)
            l2_norm_incorrect = torch.linalg.norm(icrt_im_txt_distance, dim=-1)
            #print("correct image text l2 distance shape:",l2_norm_correct.shape) 
            #print("correct image text l2 distance:",l2_norm_correct)                        
            #print("incorrect image text l2 distance shape:",l2_norm_incorrect.shape)
            #print("incorrect image text l2 distance:",l2_norm_incorrect)
            l2_norm_correct_ = l2_norm_correct.mean()
            l2_norm_incorrect_ = l2_norm_incorrect.mean()
            #print("correct image text l2-- distance shape:",l2_norm_correct_.shape) 
            #print("correct image text l2-- distance:",l2_norm_correct_)
            #print("incorrect image text l2 distance shape:",l2_norm_incorrect_.shape)
            #print("incorrect image text l2 distance:",l2_norm_incorrect_)

            self.l2_norm_correct_training = l2_norm_correct_
            self.l2_norm_incorrect_training = l2_norm_incorrect_
            #print("correct image text self l2 distance shape:",self.l2_norm_correct_training.shape)                       
            #print("incorrect image text self l2 distance shape:", self.l2_norm_incorrect_training.shape)


            #-----------------------------------------------------     
            #for cons in self.constant_list:
            #print('current feature constant:', cons)
            #getting text disperssion
            prompt_mean = text_features.mean(0)
            #print("shape of prompt mean:",prompt_mean.shape) >> shape of prompt mean: torch.Size([512])
            #print("shape of text features:",text_features.shape) >> shape of text features: torch.Size([47, 512])
            feature_distance = text_features - prompt_mean
            #print("feature distance text:",feature_distance.shape)
            #getting image disperssion
            image_mean = image_features.mean(0)
            im_feature_distance = image_features - image_mean
            #print("image feature shape:", image_features.shape)
            #print("Text features shape:", text_features.shape)
            
            #print("feature Distance: ", feature_distance)
            #distance have to be adjust not text -features
            #self.feature_distance = feature_distance
            l2_norm_image = torch.linalg.norm(im_feature_distance, dim=-1)
            l2_norm = torch.linalg.norm(feature_distance, dim=-1)
            l2_norm_mean_im = l2_norm_image.mean()
            l2_norm_mean = l2_norm.mean()
            
            #for saving to csv file
            self.l2_norm_mean = l2_norm_mean.item()
            self.l2_norm_mean = l2_norm_mean_im.item()

            #for training
            self.l2_norm_mean_training = l2_norm_mean
            self.l2_norm_mean_training_im = l2_norm_mean_im
            
            #calculating the cosine similarity 
            """similarity_im_txt = image_features @ text_features.t()
            #calculating norm of similarity
            l2_norm_similarity = torch.linalg.norm(similarity_im_txt, dim=-1)
            l2_norm_similarity_mean =  l2_norm_similarity.mean() 
            self.l2_norm_similarity_training =l2_norm_similarity_mean"""

            #tp - Ip

            """image_features_norm_l2 = torch.linalg.norm(image_features, dim=-1)
            mean_im_l2 = image_features_norm_l2.mean()
            text_features_norm_l2 = torch.linalg.norm(text_features, dim=-1)
            mean_text_l2 = text_features_norm_l2.mean()
            difference =  mean_im_l2 - mean_text_l2
            print("shape of image norm:",mean_im_l2)
            print("shape of text norm:",mean_text_l2 )
            print(" difference norm:",difference)
            self.im_txt_diff = torch.linalg.vector_norm(difference)"""
            #print("text_features shape:",text_features.shape)
            #print("image features shape:",image_features.shape)

            #batch method

            im_feature_expand = image_features.unsqueeze(1)  # Shape: (64, 1, 512)
            txt_features_expand = text_features.unsqueeze(0)  # Shape: (1, 47, 512)
            # Calculate the differences
            diff = im_feature_expand - txt_features_expand   # Shape: (64, 47, 512)
            # Calculate the Frobenius norm (equivalent to L2 norm for vectors)
            l2_norms_im_txt_ = torch.norm(diff, p=2, dim=2)
            #l2_norms_im_txt_ = torch.linalg.matrix_norm(diff, ord='fro', dim=(2,))  # Shape: (64, 47)
            l2_mt_txt_im = torch.linalg.norm(l2_norms_im_txt_, dim=-1)
            self.im_txt_diff = l2_mt_txt_im.mean()

            """    
            #loop method
            l2_norms_txt_im = torch.zeros(image_features.size(0),text_features.size(0)).to(device=args.gpu)

            for i in range(image_features.size(0)):
                for j in range(text_features.size(0)):
                    diff = image_features[i] - text_features[j]
                    l2_norms_txt_im[i, j] = torch.linalg.vector_norm(diff)
            #print("combine matrix shape:",l2_norms_txt_im.shape)        
            l2_mt_txt_im = torch.linalg.norm(l2_norms_txt_im, dim=-1)
            #print("mean value after l2 norm:",l2_mt_txt_im.mean())
            self.im_txt_diff = l2_mt_txt_im.mean()"""        

            #calculating the similarity between centroid
            l2_text_centroid = torch.linalg.norm(prompt_mean, dim=-1)
            l2_text_centroid_mean = l2_text_centroid.mean()
            l2_image_centroid = torch.linalg.norm(image_mean, dim=-1)
            l2_image_centroid_mean = l2_image_centroid.mean()
            #print("image centroid:", l2_image_centroid_mean)
            #print("text centroid:", l2_text_centroid_mean)
            #cen_sim =  l2_image_centroid_mean - l2_text_centroid_mean 
            #self.centroid_sim = torch.linalg.vector_norm(cen_sim)
            #print("centroid:",self.centroid_sim) 
        #-----------------------------------------------------

        logit_scale = self.logit_scale.exp()
        logits = logit_scale * image_features @ text_features.t()

        return logits

    def forward(self, input,cons,args):
        if isinstance(input, Tuple):
            view_0, view_1, view_2 = input
            return self.contrast_prompt_tuning(view_0, view_1, view_2)
        elif len(input.size()) == 2:
            return self.directional_prompt_tuning(input)
        else:
            return self.inference(input,cons,args)


def get_coop(clip_arch, test_set, device, n_ctx, ctx_init, constant, learned_cls=False):
    if test_set in fewshot_datasets:
        classnames = eval("{}_classes".format(test_set.lower()))
    elif test_set == 'bongard':
        if learned_cls:
            classnames = ['X', 'X']
        else:
            classnames = ['True', 'False']
    else:
        classnames = imagenet_classes

    model = ClipTestTimeTuning(device, classnames,constant, None, arch=clip_arch,
                            n_ctx=n_ctx, ctx_init=ctx_init, learned_cls=learned_cls)

    return model

