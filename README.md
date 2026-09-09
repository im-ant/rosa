# Using Reward Uncertainty to Induce Diverse Behaviour in Reinforcement Learning


[Anthony GX-Chen](https://im-ant.github.io/)<sup>\*1†</sup>, [Ankit Anand](https://sites.google.com/view/ankitsanand/home)<sup>\*2</sup>, [Gheorghe Comanici](https://scholar.google.com/citations?user=CiUedsUAAAAJ)<sup>\*2</sup>, [Zaheer Abbas](https://scholar.google.com/citations?user=i27Wt-cAAAAJ)<sup>2</sup>, [Eser Aygün](https://eseraygun.com/)<sup>2</sup>, [David Smalling](https://www.linkedin.com/in/david-smalling-9813314)<sup>2</sup>, [Shibl Mourad](https://scholar.google.com/citations?user=1UqGNJIAAAAJ)<sup>2</sup>, [Doina Precup](https://www.cs.mcgill.ca/~dprecup/)<sup>2</sup>, [André Barreto](https://sites.google.com/view/andrebarreto)<sup>\*2</sup>, [Mark Rowland](https://sites.google.com/view/markrowland)<sup>\*2</sup>

<sup>1</sup>New York University · <sup>2</sup>Google DeepMind  
<sup>\*</sup>Core contributors · <sup>†</sup>Work done as a student researcher at Google DeepMind

**[Blog post](https://im-ant.github.io/rosa/) · [arXiv](https://arxiv.org/abs/2606.03962)**

## TL;DR

We should not ask for policy diversity directly. Instead, we should characterize our
uncertainty in the reward function and train a policy that calibrates its action
probabilities to that uncertainty.

Our method, ROSA, replaces the scalar reward with a *distribution over reward functions* and applies
a *nonlinear set function* (e.g. the max) over a group of sampled actions. The resulting
optimal policy is diverse along the dimensions where the reward is uncertain, and
deterministic where it is not. In practice it is a single change to how the advantage is
computed and drops into a standard RL post-training pipeline.

The [blog post](https://im-ant.github.io/rosa/) walks through the objective with
interactive figures.

## Code

> **Status: in preparation.** 

ROSA is a simple change to the advantage calculation when multiple reward functions can be sampled or evaluated. The [blog post](https://im-ant.github.io/rosa/#estimator) provides one implementation.

## Citation

```bibtex
@misc{gxchen2026rosa,
  title={Using Reward Uncertainty to Induce Diverse Behaviour
         in Reinforcement Learning},
  author={Anthony GX-Chen and Ankit Anand and Gheorghe Comanici
          and Zaheer Abbas and Eser Ayg{\"u}n and David Smalling
          and Shibl Mourad and Doina Precup and Andr{\'e} Barreto
          and Mark Rowland},
  year={2026},
  eprint={2606.03962},
  archivePrefix={arXiv},
  primaryClass={cs.LG},
  url={https://arxiv.org/abs/2606.03962}
}
```

## License

To be added.
