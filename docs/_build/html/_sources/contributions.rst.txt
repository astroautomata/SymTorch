Contributing to SymTorch
========================

We welcome contributions to SymTorch! This guide will help you get started.

Getting Started
---------------

1. **Fork and Clone**

   Fork the repository and clone it locally::

       git clone https://github.com/elizabethsztan/SymTorch.git
       cd SymTorch

2. **Set Up Development Environment**

   Create and activate a virtual environment::

       python -m venv symtorch_venv
       source symtorch_venv/bin/activate  # On Windows: symtorch_venv\Scripts\activate

   Install the package in editable mode::

       pip install -e SymTorch/

Development Guidelines
----------------------

Code Quality
~~~~~~~~~~~~

- **Avoid code duplication**: Reuse existing helper methods rather than duplicating patterns
- **Use class constants**: Leverage ``DEFAULT_SR_PARAMS`` and helper methods like ``_create_sr_params()``
- **Follow inheritance patterns**: Extend parent functionality rather than duplicating code
- **Keep methods focused**: Each method should have a single, clear responsibility

Testing
~~~~~~~

Before submitting a pull request, ensure all tests pass::

    python -m pytest SymTorch/tests/

Key tests to verify:

- **Import functionality**: ``SymbolicMLP``, ``SymbolicModel``, ``SLIMEModel``, and ``PruningMLP`` import successfully
- **Symbolic regression pipeline**: ``distill()`` � ``switch_to_equation()`` � ``forward()`` works correctly
- **Save/load functionality**: Models save and load with preserved state

You can also run the demo notebooks to verify functionality::

    cd SymTorch/docs/demos
    jupyter notebook getting_started_demo.ipynb

Making Contributions
--------------------

Types of Contributions
~~~~~~~~~~~~~~~~~~~~~~

- **Bug fixes**: Fix issues in existing functionality
- **New features**: Add new capabilities (discuss in an issue first)
- **Documentation**: Improve docs, docstrings, or examples
- **Tests**: Add or improve test coverage
- **Examples**: Create new demo notebooks or case studies

Pull Request Process
~~~~~~~~~~~~~~~~~~~~~

1. **Create a branch** for your changes::

       git checkout -b feature/your-feature-name

2. **Make your changes** following the code quality guidelines

3. **Test your changes** thoroughly

4. **Commit your changes** with clear, descriptive messages::

       git commit -m "Add feature: description of your changes"

5. **Push to your fork**::

       git push origin feature/your-feature-name

6. **Open a pull request** on GitHub with:

   - Clear description of changes
   - Reference to any related issues
   - Test results demonstrating functionality

Code Architecture
-----------------

When to Use Each Component
~~~~~~~~~~~~~~~~~~~~~~~~~~

- **SymbolicMLP**: Layer-level symbolic regression for understanding individual MLP layers
- **SymbolicModel**: Model-level symbolic regression for end-to-end approximation
- **SLIMEModel**: Local interpretability around specific data points (model-agnostic)
- **PruningMLP**: Extension of SymbolicMLP with dimension pruning

Common Patterns to Avoid
~~~~~~~~~~~~~~~~~~~~~~~~

- Don't duplicate default parameter blocks (use ``_create_sr_params()``)
- Don't manually implement hook registration/removal (use ``_capture_layer_output()``)
- Don't copy-paste variable extraction logic (use helper methods)
- Don't override parent methods with full reimplementation

Questions or Issues?
--------------------

- **Bug reports**: Open an issue on GitHub with a minimal reproducible example
- **Feature requests**: Open an issue to discuss before implementing
- **Questions**: Check existing issues or open a new discussion

Thank you for contributing to SymTorch!
