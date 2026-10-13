import sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import json,unittest
from pathlib import Path
import torch
from cpu_bayes import BayesianCNN,BayesianLinear,uncertainty

class VariationalTests(unittest.TestCase):
    def test_posterior_sampling_does_not_mutate_batchnorm(self):
        torch.set_num_threads(1);torch.manual_seed(42)
        model=BayesianCNN();model.train();model(torch.randn(4,3,32,32))
        before={n:b.clone() for n,b in model.named_buffers()}
        mean,std=uncertainty(model,torch.randn(4,3,32,32),4)
        self.assertEqual(tuple(mean.shape),(4,10))
        self.assertTrue(torch.allclose(mean.sum(-1),torch.ones(4),atol=1e-6))
        self.assertGreater(float(std.mean()),0)
        for n,b in model.named_buffers():self.assertTrue(torch.equal(b,before[n]))
    def test_standard_normal_posterior_has_zero_kl(self):
        layer=BayesianLinear(3,2)
        with torch.no_grad():
            layer.mu.zero_();layer.bias_mu.zero_()
            inverse=torch.log(torch.expm1(torch.tensor(1.)))
            layer.rho.fill_(inverse);layer.bias_rho.fill_(inverse)
        self.assertAlmostEqual(float(layer.kl()),0,places=5)
    def test_original_notebook_keeps_bn_eval_during_mc(self):
        notebook=json.loads((Path(__file__).resolve().parents[1]/"ComputerVision.ipynb").read_text())
        code='\n'.join(''.join(c['source']) for c in notebook['cells'] if c['cell_type']=='code')
        self.assertIn('model(x, training=False, mc_dropout=True)',code)
        compile(code,'ComputerVision.ipynb','exec')

if __name__=='__main__':unittest.main()
