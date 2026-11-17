// A real response from POST /ask (agentic RAG), captured 2026-11-17.
// Shown before the first question so visitors see a full run without waiting.
import type { AskResponse } from './api'

export const EXAMPLE_QUESTION = "How do the Adam optimizer settings of CheXNet and the Transformer differ?"

export const EXAMPLE: AskResponse = {
  "answer": "Based on the provided documents, the Adam optimizer settings differ in their parameters:\n\n* **CheXNet** uses standard parameters for the Adam optimizer: $\\beta_1 = 0.9$ and $\\beta_2 = 0.999$ [1].\n* **The Transformer** uses the Adam optimizer with $\\beta_1 = 0.9$, $\\beta_2 = 0.98$, and $\\epsilon = 10^{-9}$ [2].",
  "sources": [
    "chexnet.pdf p.2",
    "transformer.pdf p.7"
  ],
  "strategy": "agentic_rag",
  "route": "comparison",
  "retries": 1,
  "latency_s": 6.83,
  "steps": [
    {
      "step": "route",
      "label": "comparison",
      "ms": 13
    },
    {
      "step": "search",
      "query": "How do the Adam optimizer settings of CheXNet and the Transformer differ?",
      "found": 8,
      "ms": 10
    },
    {
      "step": "check",
      "sufficient": false,
      "relevant": 2,
      "missing": "The provided text contains information about CheXNet's Adam optimizer settings, but lacks any information regarding the Transformer's Adam optimizer settings.",
      "ms": 1932
    },
    {
      "step": "rewrite",
      "query": "We trained the transformer model using the Adam optimizer with $\\beta_1 = 0.9$, $\\beta_2 = 0.98$ and $\\epsilon = 10^{-9}$",
      "ms": 684
    },
    {
      "step": "search",
      "query": "We trained the transformer model using the Adam optimizer with $\\beta_1 = 0.9$, $\\beta_2 = 0.98$ and $\\epsilon = 10^{-9}$",
      "found": 8,
      "ms": 594
    },
    {
      "step": "check",
      "sufficient": true,
      "relevant": 2,
      "missing": "",
      "ms": 2132
    },
    {
      "step": "answer",
      "cited": 2,
      "ms": 1470
    }
  ],
  "citations": [
    {
      "n": 1,
      "source": "chexnet.pdf",
      "page": 2,
      "text": "−w−· (1−y) logp(Y = 0|X), where p(Y =i|X) is the probability that the network assigns to the labeli,w+ =|N|/(|P|+|N|), andw− = |P|/(|P|+|N|) with|P| and|N| the number of positive cases and negative cases of pneumonia in the training set respectively. 2.2. Model Architecture and T raining CheXNet is a 121-layer Dense Convolutional Net- work (DenseNet) (Huang et al., 2016) trained on the ChestX-ray 14 dataset. DenseNets improve ﬂow of in- formation and gradients through the network, making the optimization of very deep networks tractable. We replace the ﬁnal fully connected layer with one that has a single output, after which we apply a sigmoid nonlinearity. The weights of the network are initialized with weights from a model pretrained on ImageNet (Deng et al., 2009). The network is trained end-to-end using Adam with standard parameters (β1 = 0.9 and β2 = 0.999) (Kingma & Ba, 2014). We train the model using mini- batches of size 16. We use an initial learning rate of"
    },
    {
      "n": 2,
      "source": "transformer.pdf",
      "page": 7,
      "text": "We trained our models on one machine with 8 NVIDIA P100 GPUs. For our base models using the hyperparameters described throughout the paper, each training step took about 0.4 seconds. We trained the base models for a total of 100,000 steps or 12 hours. For our big models,(described on the bottom line of table 3), step time was 1.0 seconds. The big models were trained for 300,000 steps (3.5 days). 5.3 Optimizer We used the Adam optimizer [20] with β1 = 0.9, β2 = 0.98 and ϵ = 10−9. We varied the learning rate over the course of training, according to the formula: lrate = d−0.5model · min(step_num−0.5, step_num · warmup_steps−1.5) (3) This corresponds to increasing the learning rate linearly for the first warmup_steps training steps, and decreasing it thereafter proportionally to the inverse square root of the step number. We used warmup_steps = 4000. 5.4 Regularization We employ three types of regularization during training: 7"
    }
  ]
}
