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

*SymTorch* can be installed from `PyPI <https://pypi.org/project/torch-symbolic/>`_.

.. code-block:: bash

   pip install torch-symbolic

To view or install the most recent version of SymTorch, please see our `GitHub <https://github.com/elizabethsztan/SymTorch>_`.

Overview
========
SymTorch combines PyTorch (neural networks) with PySR (symbolic regression) to automatically extract human-readable formulas from trained models. Instead of treating models as black boxes, it reveals the underlying mathematical relationships they've discovered.

1. Layer-Level Analysis (``SymbolicMLP``)
-----------------------------------------
- Wraps individual MLP layers within larger deep learning models 
- Discovers symbolic equations approximating the behavior of individual layers 
- Can switch between the MLP and the symbolic model during both forward pass and model training
- See the :doc:`SymbolicMLP Demo <demos/getting_started_demo>` for usage information

2. Model-Level Analysis (``SymbolicModel``)
-------------------------------------------
- Wraps entire PyToch models end-to-end 
- Approximates the behavior of entire models with a symbolic model 

3. Local Interpretability (``SLIMEModel``)
------------------------------------------
- Model-agnostic approach to approximating model behavior around a single point with a symbolic surrogate model 
- Symbolic extension of Local Interpretable Model-Agnostic Explanations (LIME)
- See the :doc:`SLIMEModel Demo <demos/slime_api_demo>` for usage information.

4. Reducing Model Flexibility (``PruningMLP``)
----------------------------------------------
- Automatically identifies and removes less important output dimensions on a MLP layer-level basis
- May encourage models to learn simpler and more interpretable patterns 
- See the :doc:`PruningMLP Demo <demos/toolkit_demo>` for usage information


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

.. toctree::
   :maxdepth: 1
   :caption: Development:

   release_notes
   contributions
   show_your_work

   


