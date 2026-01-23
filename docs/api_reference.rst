API Reference
=============

This section contains the complete API reference for SymTorch. Please see the :doc:`API Demos<api_demos.rst>` for examples on how to use each object or function.


SymbolicModel
~~~~~~~~~~~~~

.. autoclass:: symtorch.SymbolicModel
   :members:
   :undoc-members:
   :show-inheritance:

PySR Parameters
~~~~~~~~~~~~~~~

The :meth:`.distill` method parses in parameters to a `PySRRegressor` class. Please see `PySR <https://ai.damtp.cam.ac.uk/pysr/api/>`_ for more details.\
The default PySR parameters are

.. table:: Default configurations
   :name: tab-def-sr-params

   =======================  ==============================
   SR Parameter             Configuration
   =======================  ==============================
   Binary operators         ``+, *``
   Unary operators          ``inv(x)=1/x, sin, exp``
   Extra sympy mappings     ``"inv": lambda x: 1/x``
   Number of iterations     400
   Complexity of operators  ``sin: 3, exp: 3``
   =======================  ==============================

Saving and Loading
~~~~~~~~~~~~~~~~~~

SymbolicModel supports PyTorch's standard save/load mechanisms:

.. code-block:: python
   :linenos:
   
   # Save model state (recommended)
   torch.save(model.state_dict(), 'model.pth')

   # Load model state
   model = SymbolicModel(architecture, block_name='my_model')
   model.load_state_dict(torch.load('model.pth'))

   # Full model save/load also works
   torch.save(model, 'full_model.pth')
   model = torch.load('full_model.pth', weights_only=False)


