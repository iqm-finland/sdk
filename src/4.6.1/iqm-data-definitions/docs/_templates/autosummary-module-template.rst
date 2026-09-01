{{ name | escape | underline}}

.. automodule:: {{ fullname }}

   Full path: {{ fullname }}

   {% if '_pb2' in name %}
   .. literalinclude:: {% autoescape false %}../../protos/{{ fullname  | replace(".", "/") | replace("_pb2", "")}}.proto{% endautoescape %}
      :caption: Protobuf source code
      :language: proto
   {% endif %}

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
