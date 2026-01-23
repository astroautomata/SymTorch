Welcome to the SymTorch Documentation
=====================================

.. image:: _static/symtorch_logo.png
   :alt: SymTorch logo. 
   :width: 100%

.. image:: _static/conceptual_pic.png
   :alt: Visual example of SymbolicMLP. 
   :width: 100%

**SymTorch** is an interpretability toolkit that uses symbolic regression to reveal the behaviour of black-box models.

Installation
============

*SymTorch* will soon be released on `PyPI <https://pypi.org/project/torch-symbolic/>`_.

To view or install the most recent version of SymTorch, please see our `GitHub <https://github.com/elizabethsztan/SymTorch>`_.

Overview
========
SymTorch combines PyTorch (neural networks) with PySR (symbolic regression) to automatically extract human-readable formulas from trained models. Instead of treating models as black boxes, it reveals the underlying mathematical relationships they've discovered.

The ``SymbolicModel`` Class
----------------------------
All functionality is accessed through the unified ``SymbolicModel`` class, which supports four operational modes:

1. **Layer-Level Mode**
   - Wraps individual PyTorch layers within larger deep learning models
   - Discovers symbolic equations approximating the behavior of specific layers
   - Can switch between the original layer and symbolic equations during forward pass and training
   - See the :doc:`Getting Started Demo <demos/api_demos/getting_started_demo>`

2. **Model-Agnostic Mode**
   - Works with any callable function (PyTorch, TensorFlow, scikit-learn, or pure Python)
   - Approximates end-to-end model behavior with symbolic equations
   - Framework-independent approach to symbolic regression
   - Activated by passing a callable function to the constructor
   - See the :doc:`Getting Started Demo <demos/api_demos/getting_started_demo>`

3. **SLIME Mode** (Local Interpretability)
   - Model-agnostic approach to approximating behavior around specific data points
   - Symbolic extension of Local Interpretable Model-Agnostic Explanations (LIME)
   - Provides more expressive local explanations than linear models
   - Activated by passing ``SLIME=True`` and ``slime_params`` to ``distill()``
   - See the :doc:`SLIME Demo <demos/api_demos/slime_demo>`

4. **Pruning Capabilities**
   - Automatically identifies and removes less important output dimensions
   - Works on a layer-level basis during training
   - Encourages models to learn simpler and more interpretable patterns
   - Activated by calling ``setup_pruning()`` before training
   - See the :doc:`Pruning Demo <demos/api_demos/pruning_demo>`



Contents
========

.. toctree::
   :maxdepth: 1
   :caption: Documentation:

   api_reference
   api_demos

.. toctree::
   :maxdepth: 1
   :caption: Examples:

   demos/pinns_demo.ipynb
   demos/gnns_demo.ipynb
   demos/llm_maths_demo.ipynb

.. toctree::
   :maxdepth: 1
   :caption: Development:

   release_notes
   contributions
   show_your_work

   


