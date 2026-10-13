"""Actual CPU variational CNN training on real hosted CIFAR-10 examples."""
import argparse, hashlib, json, time
from pathlib import Path
import numpy as np
import requests
import torch
from torch import nn
from torch.nn import functional as F

CLASSES=['airplane','automobile','bird','cat','deer','dog','frog','horse','ship','truck']

class BayesianLinear(nn.Module):
    def __init__(self,input_features,output_features):
        super().__init__()
        self.mu=nn.Parameter(torch.empty(output_features,input_features).normal_(0,.03))
        self.rho=nn.Parameter(torch.full_like(self.mu,-5.))
        self.bias_mu=nn.Parameter(torch.zeros(output_features))
        self.bias_rho=nn.Parameter(torch.full_like(self.bias_mu,-5.))
    def forward(self,x,sample=True):
        weight=self.mu;bias=self.bias_mu
        if sample:
            weight=weight+F.softplus(self.rho)*torch.randn_like(weight)
            bias=bias+F.softplus(self.bias_rho)*torch.randn_like(bias)
        return F.linear(x,weight,bias)
    def kl(self):
        result=0
        for mu,rho in [(self.mu,self.rho),(self.bias_mu,self.bias_rho)]:
            sigma=F.softplus(rho)
            result=result+.5*torch.sum(mu.square()+sigma.square()-1-2*sigma.log())
        return result

class BayesianCNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.features=nn.Sequential(nn.Conv2d(3,32,3),nn.BatchNorm2d(32),nn.ReLU(),nn.MaxPool2d(2),nn.Conv2d(32,64,3),nn.BatchNorm2d(64),nn.ReLU(),nn.MaxPool2d(2),nn.Conv2d(64,128,3),nn.BatchNorm2d(128),nn.ReLU(),nn.MaxPool2d(2),nn.Flatten())
        self.hidden=BayesianLinear(512,128);self.output=BayesianLinear(128,10)
    def forward(self,x,sample=True,mc_dropout=False):
        x=self.features(x);x=F.relu(self.hidden(x,sample))
        x=F.dropout(x,p=.4,training=self.training or mc_dropout)
        return self.output(x,sample)
    def kl(self):return self.hidden.kl()+self.output.kl()

@torch.inference_mode()
def uncertainty(model,x,draws=8):
    if draws<2:raise ValueError('At least 2 posterior draws required')
    model.eval()
    before={name:b.clone() for name,b in model.named_buffers()}
    samples=torch.stack([model(x,sample=True,mc_dropout=True).softmax(-1) for _ in range(draws)])
    if any(not torch.equal(b,before[n]) for n,b in model.named_buffers()): raise RuntimeError('MC inference mutated BatchNorm statistics')
    means=samples.mean(0)
    if not torch.allclose(means.sum(-1),torch.ones(len(x)),atol=1e-5): raise RuntimeError('Invalid predictive distribution')
    return means,samples.std(0,unbiased=False)

def dataset(cache):
    """Download real CIFAR examples from the university-hosted HF dataset API.

    Hosted preview images may be re-encoded, so this is explicitly not a
    byte-for-byte canonical archive or full benchmark reproduction.
    """
    import concurrent.futures,io
    from PIL import Image
    cache=Path(cache);cache.mkdir(parents=True,exist_ok=True);saved=cache/'cifar-hosted-subset.npz'
    if saved.exists():
        d=np.load(saved);return {k:d[k] for k in d.files}
    dataset_id='uoft-cs/cifar10';endpoint='https://datasets-server.huggingface.co/rows'
    result={};provenance={'dataset':dataset_id,'config':'plain_text','endpoint':endpoint,'scope':'Real hosted CIFAR-10 preview exports; image encoding may differ from canonical archive.'}
    def fetch(entry):
        response=requests.get(entry['row']['img']['src'],timeout=30);response.raise_for_status()
        image=Image.open(io.BytesIO(response.content)).convert('RGB')
        if image.size!=(32,32):raise RuntimeError('Expected CIFAR32x32 images')
        return np.asarray(image,dtype=np.uint8).transpose(2,0,1),int(entry['row']['label']),hashlib.sha256(response.content).hexdigest()
    for split,count in [('train',512),('test',128)]:
        entries=[]
        for offset in range(0,count,100):
            response=requests.get(endpoint,params={'dataset':dataset_id,'config':'plain_text','split':split,'offset':offset,'length':min(100,count-offset)},timeout=30)
            response.raise_for_status();entries.extend(response.json()['rows'])
        with concurrent.futures.ThreadPoolExecutor(max_workers=6) as workers:
            images=list(workers.map(fetch,entries))
        result[split+'_images']=np.stack([x[0] for x in images]);result[split+'_labels']=np.array([x[1] for x in images]);result[split+'_indices']=np.arange(count)
        provenance[split+'_image_sha256']=[x[2] for x in images]
    np.savez_compressed(saved,**result);(cache/'provenance.json').write_text(json.dumps(provenance,indent=2));return result

def run(output,cache,epochs=2,train_size=512,test_size=128):
    if not 1<=epochs<=20 or not 32<=train_size<=512 or not 16<=test_size<=128:raise ValueError('Use epochs1..20, train32..512, test16..128')
    torch.set_num_threads(1);torch.manual_seed(42)
    out=Path(output);out.mkdir(parents=True,exist_ok=True);data=dataset(cache)
    x=torch.from_numpy(data['train_images'][:train_size]).float()/255;y=torch.from_numpy(data['train_labels'][:train_size]).long()
    test_x=torch.from_numpy(data['test_images'][:test_size]).float()/255;test_y=torch.from_numpy(data['test_labels'][:test_size]).long()
    model=BayesianCNN();before={n:p.detach().clone() for n,p in model.named_parameters()};optimizer=torch.optim.Adam(model.parameters(),lr=1e-3)
    losses=[];started=time.perf_counter()
    for epoch in range(epochs):
        model.train();order=torch.randperm(len(x));total=0.
        for start in range(0,len(x),32):
            chosen=order[start:start+32];optimizer.zero_grad()
            logits=model(x[chosen]);loss=F.cross_entropy(logits,y[chosen])+model.kl()/len(x)
            if not torch.isfinite(loss):raise RuntimeError('Non-finite variational objective')
            loss.backward();optimizer.step();total+=float(loss.detach())*len(chosen)
        losses.append(total/len(x))
    changed=sum(not torch.equal(p,before[n]) for n,p in model.named_parameters())
    if not changed:raise RuntimeError('Training did not change model parameters')
    means,std=uncertainty(model,test_x,8);accuracy=float((means.argmax(-1)==test_y).float().mean())
    torch.save(model.state_dict(),out/'model.pt');reload=BayesianCNN();reload.load_state_dict(torch.load(out/'model.pt',weights_only=True));reload.eval();model.eval()
    with torch.inference_mode():
        expected=model(test_x[:8],sample=False);actual=reload(test_x[:8],sample=False)
    error=float((expected-actual).abs().max())
    if not torch.equal(expected,actual):raise RuntimeError('Deterministic model reload changed logits')
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig,axes=plt.subplots(2,4,figsize=(12,6))
    for i,ax in enumerate(axes.flat):
        ax.imshow(data['test_images'][i].transpose(1,2,0));pred=int(means[i].argmax());ax.set_title(f'{CLASSES[pred]} / actual {CLASSES[int(test_y[i])]}\nmean std {float(std[i].mean()):.4f}',fontsize=9);ax.axis('off')
    fig.suptitle('Actual CIFAR-10 CPU posterior sampling · bounded subset',fontsize=14);fig.tight_layout();fig.savefig(out/'predictions.png',dpi=140);plt.close(fig)
    result={'dataset':'official CIFAR-10','source':'https://huggingface.co/datasets/uoft-cs/cifar10','image_source':'University dataset hosted preview exports; encoding may differ from canonical CIFAR archive','train_batch':'hosted train split','train_indices':data['train_indices'][:train_size].tolist(),'test_batch':'hosted test split','test_indices':data['test_indices'][:test_size].tolist(),'train_rows':len(x),'test_rows':len(test_x),'epochs':epochs,'posterior_draws':8,'variational_objective':losses,'heldout_subset_accuracy':accuracy,'changed_parameter_tensors':changed,'reload_max_logit_error':error,'batchnorm_unchanged_during_mc':True,'uncertainty_mean_std':float(std.mean()),'elapsed_train_seconds':time.perf_counter()-started,'scope':'Bounded hosted-CIFAR-image CPU variational reparameterization CNN; not identical TensorFlow DenseFlipout estimator, full benchmark reproduction, or calibrated uncertainty.'}
    (out/'result.json').write_text(json.dumps(result,indent=2));return result

if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--output',default='cpu-bayes-output');parser.add_argument('--cache',default='.data-cache');parser.add_argument('--epochs',type=int,default=2);parser.add_argument('--train-size',type=int,default=512);parser.add_argument('--test-size',type=int,default=128);args=parser.parse_args();print(json.dumps(run(args.output,args.cache,args.epochs,args.train_size,args.test_size),indent=2))
