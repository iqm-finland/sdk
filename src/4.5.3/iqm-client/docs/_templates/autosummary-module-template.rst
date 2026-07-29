{{ name | escape | underline}}

Full path: {{ fullname }}

.. automodule:: {{ fullname }}
   {% block attributes %}
   {% if attributes %}
   .. rubric:: Module Attributes

   .. autosummary::
      :toctree:
   {% for item in attributes %}
      {{ item }}
   {%- endfor %}
   {% endif %}
   {% endblock %}

   {% block functions %}
   {% if functions %}
   .. rubric:: {{ _('Functions') }}

   .. autosummary::
      :toctree:
   {% for item in functions %}
      {{ item }}
   {%- endfor %}
   {% endif %}
   {% endblock %}

   {% block classes %}
   {% if classes %}
   .. rubric:: {{ _('Classes') }}

   .. autosummary::
      :toctree:
      :nosignatures:
      :template: autosummary-class-template.rst
   {% for item in classes %}
      {{ item }}
   {%- endfor %}
   {% if fullname == 'iqm.iqm_server_client.iqm_server_client' %}
      _IQMServerClient
   {% endif %}
   {% endif %}
   {% endblock %}

   {% block exceptions %}
   {% if exceptions %}
   .. rubric:: {{ _('Exceptions') }}

   .. autosummary::
      :toctree:
   {% for item in exceptions %}
      {{ item }}
   {%- endfor %}
   {% endif %}
   {% endblock %}

{% block modules %}
{% if modules %}
.. rubric:: Subpackages and modules

.. autosummary::
   :toctree:
   :template: autosummary-module-template.rst
   :recursive:
{% for item in modules %}
   ~{{ item }}
{%- endfor %}
{% endif %}
{% endblock %}

{% block inheritance_diagram %}
{% if classes %}
.. rubric:: Inheritance

.. inheritance-diagram:: {{ fullname }}
   :parts: 1
   :private-bases:
{% endif %}
{% endblock %}
