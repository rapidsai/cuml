{{ objname | escape | underline(line="=") }}

{% if objtype == "module" -%}

.. automodule:: {{ fullname }}

{%- elif objtype == "function" -%}

.. currentmodule:: {{ module }}

.. autofunction:: {{ objname }}

{%- elif objtype == "class" -%}

.. currentmodule:: {{ module }}

.. autoclass:: {{ objname }}
   :members:

{%- else -%}

.. currentmodule:: {{ module }}

.. auto{{ objtype }}:: {{ objname }}

{%- endif -%}
