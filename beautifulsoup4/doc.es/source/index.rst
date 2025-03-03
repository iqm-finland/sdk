.. _manual:

=================================
 Documentación de Beautiful Soup
=================================

.. py:module:: bs4

.. image:: 6.1.jpg
   :align: right
   :alt: "El lacayo-pez empezó por sacarse de debajo del brazo una gran carta,
	 casi tan grande como él."

`Beautiful Soup <http://www.crummy.com/software/BeautifulSoup/>`_ es una
librería de Python para extraer datos de archivos en formato HTML y XML.
Trabaja con tu analizador favorito para ofrecer maneras bien definidas
de navegar, buscar y modificar el árbol analizado. Puede llegar a ahorrar
horas o días de trabajo a los programadores. 

Este manual ilustra con ejemplos la funcionalidades más importantes
de Beautiful Soup 4. Te muestro las cosas para las que la librería es buena,
cómo funciona, cómo usarla, cómo hacer lo que quieres y qué hacer cuando
no se cumplen tus expectativas.

Este documento cubre Beautiful Soup versión 4.12.1. Los ejemplos en este
documento fueron escritos para Python 3.8.

Podrías estar buscando la documentación de `Beautiful Soup 3
<http://www.crummy.com/software/BeautifulSoup/bs3/documentation.html>`_.
Si es así, debes saber que Beautiful Soup 3 ya no se desarrolla y
su soporte fue abandonado el 31 de diciembre de 2020. Si quieres
conocer la diferencias entre Beautiful Soup 3 y Beautiful Soup 4,
mira `Actualizar el código a BS4`_.

Esta documentación ha sido traducida a otras lenguas por los usuarios
de Beautiful Soup:

* `这篇文档当然还有中文版. <https://www.crummy.com/software/BeautifulSoup/bs4/doc.zh/>`_
* このページは日本語で利用できます(`外部リンク <http://kondou.com/BS4/>`_)
* `이 문서는 한국어 번역도 가능합니다. <https://www.crummy.com/software/BeautifulSoup/bs4/doc.ko/>`_
* `Este documento também está disponível em Português do Brasil. <https://www.crummy.com/software/BeautifulSoup/bs4/doc.ptbr>`_
* `Эта документация доступна на русском языке. <https://www.crummy.com/software/BeautifulSoup/bs4/doc.ru/>`_
 
Cómo conseguir ayuda
====================
Si tienes alguna pregunta sobre BeautifulSoup, o si tienes problemas,
`envía un correo electrónico al grupo de discusión
<https://groups.google.com/forum/?fromgroups#!forunm/beautifulsoup>`_.
Si tienes algún problema relacionado con el análisis de un documento HTML,
asegúrate de mencionar :ref:`lo que la función diagnose() dice <diagnose>`
sobre dicho documento.

Cuando informes de algún error en esta documentación, por favor,
indica la traducción que estás leyendo.

===============
 Inicio rápido
===============

Este es un documento HTML que usaré como ejemplo a lo largo de este
documento. Es parte de una historia de `Alicia en el país de las maravillas`::

 html_doc = """<html><head><title>The Dormouse's story</title></head>
 <body>
 <p class="title"><b>The Dormouse's story</b></p>

 <p class="story">Once upon a time there were three little sisters; and their names were
 <a href="http://example.com/elsie" class="sister" id="link1">Elsie</a>,
 <a href="http://example.com/lacie" class="sister" id="link2">Lacie</a> and
 <a href="http://example.com/tillie" class="sister" id="link3">Tillie</a>;
 and they lived at the bottom of a well.</p>

 <p class="story">...</p>
 """

Al procesar el documento de "Las tres hermanas" en Beautiful Soup, se nos
devuelve un objeto :py:class:`BeautifulSoup`, que representa el
documento como una estructura de datos anidada::

 from bs4 import BeautifulSoup
 soup = BeautifulSoup(html_doc, 'html.parser')

 print(soup.prettify())
 # <html>
 #  <head>
 #   <title>
 #    The Dormouse's story
 #   </title>
 #  </head>
 #  <body>
 #   <p class="title">
 #    <b>
 #     The Dormouse's story
 #    </b>
 #   </p>
 #   <p class="story">
 #    Once upon a time there were three little sisters; and their names were
 #    <a class="sister" href="http://example.com/elsie" id="link1">
 #     Elsie
 #    </a>
 #    ,
 #    <a class="sister" href="http://example.com/lacie" id="link2">
 #     Lacie
 #    </a>
 #    and
 #    <a class="sister" href="http://example.com/tillie" id="link3">
 #     Tillie
 #    </a>
 #    ; and they lived at the bottom of a well.
 #   </p>
 #   <p class="story">
 #    ...
 #   </p>
 #  </body>
 # </html>

Estas son algunas de las maneras sencillas para navegar
por la estructura de datos::

 soup.title
 # <title>The Dormouse's story</title>

 soup.title.name
 # u'title'

 soup.title.string
 # u'The Dormouse's story'

 soup.title.parent.name
 # u'head'

 soup.p
 # <p class="title"><b>The Dormouse's story</b></p>

 soup.p['class']
 # u'title'

 soup.a
 # <a class="sister" href="http://example.com/elsie" id="link1">Elsie</a>

 soup.find_all('a')
 # [<a class="sister" href="http://example.com/elsie" id="link1">Elsie</a>,
 #  <a class="sister" href="http://example.com/lacie" id="link2">Lacie</a>,
 #  <a class="sister" href="http://example.com/tillie" id="link3">Tillie</a>]

 soup.find(id="link3")
 # <a class="sister" href="http://example.com/tillie" id="link3">Tillie</a>

Una tarea frecuente es extraer todas las URL encontradas en las etiquetas
<a> de una página::

 for link in soup.find_all('a'):
     print(link.get('href'))
 # http://example.com/elsie
 # http://example.com/lacie
 # http://example.com/tillie

Otra tarea habitual es extraer todo el texto de una página::

 print(soup.get_text())
 # The Dormouse's story
 #
 # The Dormouse's story
 #
 # Once upon a time there were three little sisters; and their names were
 # Elsie,
 # Lacie and
 # Tillie;
 # and they lived at the bottom of a well.
 #
 # ...

¿Esto se parece a lo que necesitas? Si es así, sigue leyendo.

=========================
 Instalar Beautiful Soup
=========================
Si usas una versión reciente de Debian o Ubuntu Linux, puedes instalar
Beautiful Soup con el gestor de paquetes del sistema:

:kbd:`$ apt-get install python3-bs4`

Beautiful Soup 4 está publicado en Pypi, así que si no puedes instalarlo
con el gestor de paquetes, puedes instalarlo con ``easy_install`` o
``pip``. El nombre del paquete es ``beautifulsoup4``. Asegúrate de que
usas la versión correcta de ``pip`` o ``easy_install`` para tu versión
de Python (podrían llamarse ``pip3`` y ``easy_install3``, respectivamente):

:kbd:`$ easy_install beautifulsoup4`

:kbd:`$ pip install beautifulsoup4`

(El paquete :py:class:`BeautifulSoup` ``no`` es el que quieres. Ese es
el lanzamiento anterior `Beautiful Soup 3`_. Muchos *software* utilizan
BS3, así que aún está disponible, pero si estás escribiendo nuevo código,
deberías instalar ``beautifulsoup4``).

Si no tienes ``easy_install`` o ``pip`` instalados, puedes
`descargar el código de Beautiful Soup 4 comprimido en un tarball
<http://www.crummy.com/software/BeautifulSoup/download/4.x/>`_ e
instalarlo con ``setup.py``:

:kbd:`$ python setup.py install`

Si aún así todo falla, la licencia de Beautiful Soup te permite
empaquetar la librería completa con tu aplicación. Puedes descargar
el *tarball*, copiar su directorio ``bs4`` en tu base de código y
usar Beautiful Soup sin instalarlo en absoluto.

Yo empleo Python 3.10 para desarrollar Beautiful Soup, aunque debería
funcionar con otras versiones recientes.

.. _parser-installation:


Instalar un analizador
======================

Beautiful Soup soporta el analizador de HTML incluido en la librería
estándar de Python, aunque también soporta varios analizadores de
Python de terceros. Uno de ellos es el `analizador de lxml <http://lxml.de/>`_.
Dependiendo de tu instalación, puedes instalar lxml con uno de los
siguientes comandos:

:kbd:`$ apt-get install python-lxml`

:kbd:`$ easy_install lxml`

:kbd:`$ pip install lxml`

Otra alternativa es usar el analizador de Python de
`html5lib <http://code.google.com/p/html5lib/>`_,
el cual analiza HTML de la misma manera en la que lo haría
un navegador web. Dependiendo de tu instalación, puedes instalar
html5lib con uno de los siguientes comandos:

:kbd:`$ apt-get install python-html5lib`

:kbd:`$ easy_install html5lib`

:kbd:`$ pip install html5lib`

Esta tabla resume las ventajas e inconvenientes de cada librería de los analizadores:

+-----------------------+--------------------------------------------+-----------------------------------+-----------------------------+
| Analizador            | Uso típico                                 | Ventajas                          | Desventajas                 |
+-----------------------+--------------------------------------------+-----------------------------------+-----------------------------+
| html.parser de Python | ``BeautifulSoup(markup, "html.parser")``   | * Ya incluido                     | * No tan rápido como lxml,  |
|                       |                                            | * Rapidez decente                 |   menos tolerante que       |
|                       |                                            | * Tolerante (en Python 3.2)       |   html5lib.                 |
+-----------------------+--------------------------------------------+-----------------------------------+-----------------------------+
| Analizador HTML de    | ``BeautifulSoup(markup, "lxml")``          | * Muy rápido                      | * Dependencia externa de C  |
| lxml                  |                                            | * Tolerante                       |                             |
+-----------------------+--------------------------------------------+-----------------------------------+-----------------------------+
| Analizador XML de     | ``BeautifulSoup(markup, "lxml-xml")``      | * Muy rápido                      | * Dependencia externa de C  |
| lxml                  | ``BeautifulSoup(markup, "xml")``           | * El único analizador XML         |                             |
|                       |                                            |   actualmente soportado           |                             |
+-----------------------+--------------------------------------------+-----------------------------------+-----------------------------+
| html5lib              | ``BeautifulSoup(markup, "html5lib")``      | * Extremadamente tolerante        | * Muy lento                 |
|                       |                                            | * Analiza las páginas de la misma | * Dependencia externa de    |
|                       |                                            |   manera que un navegador web     |   Python                    |
|                       |                                            | * Crea HTML5 válido               |                             |
+-----------------------+--------------------------------------------+-----------------------------------+-----------------------------+

Si puedes, te recomiendo que instales y uses lxml para mayor velocidad.

Ten en cuenta que si un documento es inválido, analizadores diferentes
generarán árboles de Beautiful Soup diferentes para él. Mira
`Diferencias entre analizadores`_ para más detalle.

==================
 Haciendo la sopa
==================

Para analizar un documento pásalo al constructor de :py:class:`BeautifulSoup`.
Puedes pasar una cadena de caracteres o abrir un manejador de archivos::

 from bs4 import BeautifulSoup

 with open("index.html") as fp:
     soup = BeautifulSoup(fp, 'html.parser')

 soup = BeautifulSoup("<html>a web page</html>", 'html.parser')

Primero, el documento se convierte a Unicode, y las entidades HTML se
convierten a caracteres Unicode::

 print(BeautifulSoup("<html><head></head><body>Sacr&eacute; bleu!</body></html>", "html.parser"))
 # <html><head></head><body>Sacré bleu!</body></html>

Entonces Beautiful Soup analiza el documento usando el mejor analizador
disponible. Usará un analizador HTML a no ser que se especifique que se
use un analizador XML (ver `Analizar XML`_).

==================
 Tipos de objetos
==================

Beautiful Soup transforma un complejo documento HTML en un complejo árbol de objetos
de Python. Pero tan solo tendrás que lidiar con cuatro `tipos` de objetos: :py:class:`Tag`,
:py:class:`NavigableString`, :py:class:`BeautifulSoup` y :py:class:`Comment`.

.. py:class:: Tag

 Un objeto :py:class:`Tag` corresponde a una etiqueta XML o HTML en el documento
 original.

 ::

  soup = BeautifulSoup('<b class="boldest">Extremely bold</b>', 'html.parser')
  tag = soup.b
  type(tag)
  # <class 'bs4.element.Tag'>

 Las etiquetas tienen muchos atributos y métodos, y cubriré la mayoría de ellos en
 `Navegar por el árbol`_ y `Buscar en el árbol`_. Por ahora, las características
 más importantes de una etiqueta son su nombre y sus atributos.

 .. py:attribute:: name

  Toda etiqueta tiene un nombre::

   tag.name
   # 'b'


  Si cambias el nombre de una etiqueta, el cambio se verá reflejado en
  cualquier especificación generada por Beautiful Soup a partir de entonces::

   tag.name = "blockquote"
   tag
   # <blockquote class="boldest">Extremely bold</blockquote>

 .. py:attribute:: attrs

  Una etiqueta HTML o XML puede tener cualquier cantidad de atributos.
  La etiqueta ``<b id="boldest">`` tiene un atributo "id" cuyo valor
  es "boldest". Puedes acceder a los atributos de una etiqueta
  usándola como un diccionario::

   tag = BeautifulSoup('<b id="boldest">bold</b>', 'html.parser').b
   tag['id']
   # 'boldest'

  Puedes acceder a los atributos del diccionario directamente con ``.attrs``::

   tag.attrs
   # {'id': 'boldest'}

  Puedes añadir, quitar y modificar los atributos de una etiqueta. De nuevo, esto
  se realiza usando la etiqueta como un diccionario::

   tag['id'] = 'verybold'
   tag['another-attribute'] = 1
   tag
   # <b another-attribute="1" id="verybold"></b>

   del tag['id']
   del tag['another-attribute']
   tag
   # <b>bold</b>

   tag['id']
   # KeyError: 'id'
   tag.get('id')
   # None

  .. _multivalue:

  Atributos multivaluados
  -----------------------

  HTML 4 define algunos atributos que pueden tomar múltiples valores. HTML 5
  elimina un par de ellos, pero define unos cuantos más. El atributo multivaluado
  más común es ``class`` (esto es, una etiqueta puede tener más de una clase de CSS).
  Otros incluyen ``rel``, ``rev``, ``accept-charset``, ``headers`` y ``accesskey``.
  Por defecto, Beautiful Soup transforma los valores de un atributo multivaluado en
  una lista::

   css_soup = BeautifulSoup('<p class="body"></p>', 'html.parser')
   css_soup.p['class']
   # ['body']
  
   css_soup = BeautifulSoup('<p class="body strikeout"></p>', 'html.parser')
   css_soup.p['class']
   # ['body', 'strikeout']

  Si un atributo `parece` que tiene más de un valor, pero no es un atributo
  multivaluado definido como tal por ninguna versión del estándar de HTML,
  Beautiful Soup no modificará el atributo::

   id_soup = BeautifulSoup('<p id="my id"></p>', 'html.parser')
   id_soup.p['id']
   # 'my id'

  Cuando transformas una etiqueta en una cadena de caracteres, muchos atributos
  se combinan::

   rel_soup = BeautifulSoup('<p>Back to the <a rel="index first">homepage</a></p>', 'html.parser')
   rel_soup.a['rel']
   # ['index', 'first']
   rel_soup.a['rel'] = ['index', 'contents']
   print(rel_soup.p)
   # <p>Back to the <a rel="index contents">homepage</a></p>

  Puedes forzar que todos los atributos sean analizados como cadenas
  de caracteres pasando ``multi_valued_attributes=None`` como argumento
  clave en el constructor de :py:class:`BeautifulSoup`::

   no_list_soup = BeautifulSoup('<p class="body strikeout"></p>', 'html.parser', multi_valued_attributes=None)
   no_list_soup.p['class']
   # 'body strikeout'

  Puedes usar  ``get_attribute_list`` para obtener un valor que siempre sea una lista,
  sin importar si es un atributo multivaluado::

   id_soup.p.get_attribute_list('id')
   # ["my id"]
 
  Si analizas un documento como XML, no hay atributos multivaluados::

   xml_soup = BeautifulSoup('<p class="body strikeout"></p>', 'xml')
   xml_soup.p['class']
   # 'body strikeout'

  Una vez más, puedes configurar esto usando el argumento ``multi_valued_attributes`` ::

   class_is_multi= { '*' : 'class'}
   xml_soup = BeautifulSoup('<p class="body strikeout"></p>', 'xml', multi_valued_attributes=class_is_multi)
   xml_soup.p['class']
   # ['body', 'strikeout']

  Probablemente no tengas que hacer esto, pero si lo necesitas, usa los
  parámetros por defecto como guía. Implementan las reglas descritas en la
  especificación de HTML::

   from bs4.builder import builder_registry
   builder_registry.lookup('html').DEFAULT_CDATA_LIST_ATTRIBUTES
  
.. py:class:: NavigableString

-----------------------------

Un *string* corresponde a un trozo de texto en una etiqueta. Beautiful Soup usa la clase
:py:class:`NavigableString` para contener estos trozos de texto::

 soup = BeautifulSoup('<b class="boldest">Extremely bold</b>', 'html.parser')
 tag = soup.b
 tag.string
 # 'Extremely bold'
 type(tag.string)
 # <class 'bs4.element.NavigableString'>

Un :py:class:`NavigableString` es como una cadena de caracteres de Python Unicode,
exceptuando que también soporta algunas de las características descritas en
`Navegar por el árbol`_ y `Buscar en el árbol`_. Puedes convertir un objeto
:py:class:`NavigableString` a una cadena de caracteres Unicode usando ``str``::

 unicode_string = str(tag.string)
 unicode_string
 # 'Extremely bold'
 type(unicode_string)
 # <type 'str'>

No puedes editar dicha cadena, pero puedes reemplazar una cadena por otra, usando
:ref:`replace_with()`::

 tag.string.replace_with("No longer bold")
 tag
 # <b class="boldest">No longer bold</b>

:py:class:`NavigableString` soporta la mayoría de las características descritas en
`Navegar por el árbol`_ y `Buscar en el árbol`_, pero no todas.
En particular, como una cadena no puede contener nada (la manera en la que
una etiqueta contiene una cadena de caracteres u otra etiqueta), *strings* no
admiten los atributos `.contents`` o ``.string``, o el método ``find()``.

Si quieres usar un :py:class:`NavigableString` fuera de Beautiful Soup,
deberías llamar ``unicode()`` sobre él para convertirlo en una cadena de caracteres
de Python Unicode. Si no, tu cadena arrastrará una referencia a todo el árbol analizado
de Beautiful Soup, incluso cuando hayas acabado de utilizar Beautiful Soup. Esto es un
gran malgasto de memoria.

.. py:class:: BeautifulSoup

---------------------------

El objeto :py:class:`BeautifulSoup` representa el documento analizado
en su conjunto. Para la mayoría de propósitos, puedes usarlo como un objeto
:py:class:`Tag`. Esto significa que soporta la mayoría de métodos descritos
en `Navegar por el árbol`_ and `Buscar en el árbol`_.

Puedes también pasar un objeto :py:class:`BeautifulSoup` en cualquiera de
los métodos definidos en `Modificar el árbol`_, como si fuese un :py:class:`Tag`.
Esto te permite hacer cosas como combinar dos documentos analizados::

 doc = BeautifulSoup("<document><content/>INSERT FOOTER HERE</document", "xml")
 footer = BeautifulSoup("<footer>Here's the footer</footer>", "xml")
 doc.find(text="INSERT FOOTER HERE").replace_with(footer)
 # 'INSERT FOOTER HERE'
 print(doc)
 # <?xml version="1.0" encoding="utf-8"?>
 # <document><content/><footer>Here's the footer</footer></document>

Como un objeto :py:class:`BeautifulSoup` no corresponde realmente con una
etiqueta HTML o XML, no tiene nombre ni atributos. Aún así, es útil
comprobar su ``.name``, así que se le ha dado el ``.name`` especial
"[document]"::

 soup.name
 # '[document]'

Cadenas especiales
==================

:py:class:`Tag`, :py:class:`NavigableString` y
:py:class:`BeautifulSoup` cubren la mayoría de todo lo que verás en
un archivo HTML o XML, aunque aún quedan algunos remanentes. El principal
que probablemente encuentres es el :py:class:`Comment`.

.. py:class:: Comment

::

 markup = "<b><!--Hey, buddy. Want to buy a used parser?--></b>"
 soup = BeautifulSoup(markup, 'html.parser')
 comment = soup.b.string
 type(comment)
 # <class 'bs4.element.Comment'>

El objeto :py:class:`Comment` es solo un tipo especial de :py:class:`NavigableString`::

 comment
 # 'Hey, buddy. Want to buy a used parser'

Pero cuando aparece como parte de un documento HTML, un :py:class:`Comment`
se muestra con un formato especial::

 print(soup.b.prettify())
 # <b>
 #  <!--Hey, buddy. Want to buy a used parser?-->
 # </b>

Para documentos HTML
--------------------

Beautiful Soup define algunas subclases de :py:class:`NavigableString`
para contener cadenas de caracteres encontradas dentro de etiquetas
HTML específicas. Esto hace más fácil tomar el cuerpo principal de la
página, ignorando cadenas que probablemente representen directivas de
programación encontradas dentro de la página. `(Estas clases son nuevas
en Beautiful Soup 4.9.0, y el analizador html5lib no las usa)`.

.. py:class:: Stylesheet

Una subclase de :py:class:`NavigableString` que representa hojas de estilo
CSS embebidas; esto es, cualquier cadena en una etiqueta
``<style>`` durante el análisis del documento.

.. py:class:: Script

Una subclase de :py:class:`NavigableString` que representa
JavaScript embebido; esto es, cualquier cadena en una etiqueta
``<script>`` durante el análisis del documento.

.. py:class:: Template

Una subclase de :py:class:NavigableString` que representa plantillas
HTML embebidas; esto es, cualquier cadena en una etiqueta ``<template>``
durante el análisis del documento.

Para documentos XML
-------------------

Beautiful Soup define algunas clases :py:class:`NavigableString`
para contener tipos especiales de cadenas de caracteres que pueden
ser encontradas en documentos XML. Como :py:class:`Comment`, estas
clases son subclases de :py:class:`NavigableString` que añaden
algo extra a la cadena de caracteres en la salida.

.. py:class:: Declaration

Una subclase de :py:class:`NavigableString` que representa la
`declaración <https://www.w3.org/TR/REC-xml/#sec-prolog-dtd>`_ al
principio de un documento XML.

.. py:class:: Doctype

Una subclase de :py:class:`NavigableString` que representa la
`declaración del tipo de documento <https://www.w3.org/TR/REC-xml/#dt-doctype>`_
que puede encontrarse cerca del comienzo de un documento XML.

.. py:class:: CData

Una subclase de :py:class:`NavigableString` que representa una
`sección CData <https://www.w3.org/TR/REC-xml/#sec-cdata-sect>`_.

.. py:class:: ProcessingInstruction

Una subclase de :py:class:`NavigableString` que representa el contenido de
una `instrucción de procesamiento XML <https://www.w3.org/TR/REC-xml/#sec-pi>`_.


======================
 Navegar por el árbol
======================

Aquí está el documento HTML de las "Tres hermanas" de nuevo::

 html_doc = """
 <html><head><title>The Dormouse's story</title></head>
 <body>
 <p class="title"><b>The Dormouse's story</b></p>

 <p class="story">Once upon a time there were three little sisters; and their names were
 <a href="http://example.com/elsie" class="sister" id="link1">Elsie</a>,
 <a href="http://example.com/lacie" class="sister" id="link2">Lacie</a> and
 <a href="http://example.com/tillie" class="sister" id="link3">Tillie</a>;
 and they lived at the bottom of a well.</p>

 <p class="story">...</p>
 """

 from bs4 import BeautifulSoup
 soup = BeautifulSoup(html_doc, 'html.parser')

Usaré este como ejemplo para enseñarte cómo mover una parte de un
documento a otra.

Bajar
=====

Las etiquetas pueden contener cadenas u otras etiquetas. Estos elementos
son los hijos (`children`) de la etiqueta. Beautiful Soup ofrece muchos
atributos para navegar e iterar por los hijos de una etiqueta.

Debe notarse que las cadenas de Beautiful Soup no soportan ninguno
de estos atributos, porque una cadena no puede tener hijos.

Navegar usando nombres de etiquetas
-----------------------------------

La manera más simple de navegar por el árbol analizado es indicar
el nombre de la etiqueta que quieres. Si quieres la etiqueta <head>,
tan solo indica ``soup.head``::

 soup.head
 # <head><title>The Dormouse's story</title></head>

 soup.title
 # <title>The Dormouse's story</title>

Puedes usar este truco una y otra vez para acercarte a una parte concreta
del árbol analizado. Este código obtiene la primera etiqueta <b> dentro
de la etiqueta <body>::

 soup.body.b
 # <b>The Dormouse's story</b>

Usar el nombre de la etiqueta como atributo te dará solo la `primera`
etiqueta con ese nombre::

 soup.a
 # <a class="sister" href="http://example.com/elsie" id="link1">Elsie</a>

Si necesitas obtener `todas` las etiquetas <a>, o cualquier
cosa más complicada que la primera etiqueta con cierto nombre, tendrás
que usar uno de los métodos descritos en `Buscar en el árbol`_, como
`find_all()`::

 soup.find_all('a')
 # [<a class="sister" href="http://example.com/elsie" id="link1">Elsie</a>,
 #  <a class="sister" href="http://example.com/lacie" id="link2">Lacie</a>,
 #  <a class="sister" href="http://example.com/tillie" id="link3">Tillie</a>]

``.contents`` y ``.children``
-----------------------------

Los hijos de una etiqueta están disponibles en una lista llamada
``.contents``::

 head_tag = soup.head
 head_tag
 # <head><title>The Dormouse's story</title></head>

 head_tag.contents
 # [<title>The Dormouse's story</title>]

 title_tag = head_tag.contents[0]
 title_tag
 # <title>The Dormouse's story</title>
 title_tag.contents
 # ['The Dormouse's story']

El objeto :py:class:`BeautifulSoup` por sí solo ya tiene hijos. En este caso,
la etiqueta <html> is hija del objeto :py:class:`BeautifulSoup`.::

 len(soup.contents)
 # 1
 soup.contents[0].name
 # 'html'

Una cadena no tiene ``.contents``, porque no puede contener nada::

 text = title_tag.contents[0]
 text.contents
 # AttributeError: 'NavigableString' object has no attribute 'contents'

En lugar de obtenerlos como una lista, puedes iterar sobre los hijos
de una etiqueta usando el generador ``.children``::

 for child in title_tag.children:
     print(child)
 # The Dormouse's story

Si quieres modificar los hijos de una etiqueta, emplea los métodos
descritos en `Modificar el árbol`_. No modifiques la lista
``.contents`` directamente: eso podría ocasionar problemas que pueden
ser sutiles y difíciles de detectar.

 
``.descendants``
----------------

Los atributos ``.contents`` y ``.children`` tan solo consideran los
hijos `directos` de una etiqueta. Por ejemplo, la etiqueta <head>
tiene un único hijo directo--la etiqueta <title>::

 head_tag.contents
 # [<title>The Dormouse's story</title>]

Pero la etiqueta <title> tiene un hijo: la cadena "The Dormouse's
story". Puede dar la sensación de que esa cadena es también hija de
la etiqueta <head>. El atributo ``.descendants`` te permite iterar
sobre `todos` los hijos de una etiqueta recursivamente: sus hijos,
hijos de sus hijos directos, y así sucesivamente::

 for child in head_tag.descendants:
     print(child)
 # <title>The Dormouse's story</title>
 # The Dormouse's story

La etiqueta <head> tiene un solo hijo, pero tiene dos descendientes:
la etiqueta <title> y el hijo de la etiqueta <title>. El objeto
:py:class:`BeautifulSoup` tiene un hijo directo (la etiqueta <html>), pero
tiene otros muchos descendientes::

 len(list(soup.children))
 # 1
 len(list(soup.descendants))
 # 26

.. _.string:

``.string``
-----------

Si una etiqueta tiene solo un hijo, y dicho hijo es un :py:class:`NavigableString`,
el hijo se obtiene mediante ``.string``::

 title_tag.string
 # 'The Dormouse's story'

Si el único hijo de una etiqueta es otra etiqueta, y `esa`
etiqueta tiene un ``.string``, entonces se considera que
la etiqueta madre tiene el mismo ``.string`` que su hijo::

 head_tag.contents
 # [<title>The Dormouse's story</title>]

 head_tag.string
 # 'The Dormouse's story'

Si una etiqueta contiene más una cadena, entonces no está claro
a qué se debería referir ``.string``, así que ``.string``
pasa a valer ``None``::

 print(soup.html.string)
 # None

.. _string-generators:

``.strings`` y ``stripped_strings``
-----------------------------------

Si hay más de una cosa dentro de una etiqueta, puedes seguir
obteniendo las cadenas. Usa el generador ``.string``::

 for string in soup.strings:
     print(repr(string))
     '\n'
 # "The Dormouse's story"
 # '\n'
 # '\n'
 # "The Dormouse's story"
 # '\n'
 # 'Once upon a time there were three little sisters; and their names were\n'
 # 'Elsie'
 # ',\n'
 # 'Lacie'
 # ' and\n'
 # 'Tillie'
 # ';\nand they lived at the bottom of a well.'
 # '\n'
 # '...'
 # '\n'

Estas cadenas tienden a tener muchos espacios en blanco extra, los
cuales puedes quitar usando el generador ``.stripped_strings``::

 for string in soup.stripped_strings:
     print(repr(string))
 # "The Dormouse's story"
 # "The Dormouse's story"
 # 'Once upon a time there were three little sisters; and their names were'
 # 'Elsie'
 # ','
 # 'Lacie'
 # 'and'
 # 'Tillie'
 # ';\n and they lived at the bottom of a well.'
 # '...'

Aquí, las cadenas que consisten completamente en espacios en blanco
se ignoran, y espacios en blanco al principio y final de las cadenas
se eliminan.

Subir
=====

Continuando con la analogía del árbol genealógico, toda etiqueta
tiene una `madre`: la etiqueta que la contiene.

.. _.parent:

``.parent``
-----------

Puedes acceder a la madre de una etiqueta con el atributo ``.parent``. En
el ejemplo de "Las tres hermanas", la etiqueta <head> es la madre
de la etiqueta <title>::

 title_tag = soup.title
 title_tag
 # <title>The Dormouse's story</title>
 title_tag.parent
 # <head><title>The Dormouse's story</title></head>

El texto de título tiene una madre: la etiqueta <title> que lo
contiene::

 title_tag.string.parent
 # <title>The Dormouse's story</title>

La madre de una etiqueta de alto nivel como <html> es el objeto :py:class:`BeautifulSoup`
mismo::

 html_tag = soup.html
 type(html_tag.parent)
 # <class 'bs4.BeautifulSoup'>

Y el ``.parent`` de un objeto :py:class:`BeautifulSoup` se define como ``None``::

 print(soup.parent)
 # None

.. _.parents:

``.parents``
------------

Puedes iterar sobre todas las madres de los elementos con
``.parents``. Este ejemplo usa ``.parent`` para moverse' de una
etiqueta <a> en medio del documento a lo más alto del documento::

 link = soup.a
 link
 # <a class="sister" href="http://example.com/elsie" id="link1">Elsie</a>
 for parent in link.parents:
     print(parent.name)
 # p
 # body
 # html
 # [document]

Hacia los lados
===============

Considera un documento sencillo como este::

 sibling_soup = BeautifulSoup("<a><b>text1</b><c>text2</c></a>", 'html.parser')
 print(sibling_soup.prettify())
 #   <a>
 #    <b>
 #     text1
 #    </b>
 #    <c>
 #     text2
 #    </c>
 #   </a>

Las etiquetas <b> y <c> están al mismo nivel: son hijas directas de la misma
etiqueta. Las llamamos `hermanas`. Cuando un documento está bien formateado,
las hermanas están al mismo nivel de sangría. Puedes usar también esta
relación en el código que escribas.

``.next_sibling`` y ``.previous_sibling``
-----------------------------------------

Puedes usar ``.next_sibling`` y ``.previous_sibling`` para navegar
entre elementos de la página que están al mismo nivel del árbol
analizado::

 sibling_soup.b.next_sibling
 # <c>text2</c>

 sibling_soup.c.previous_sibling
 # <b>text1</b>

La etiqueta <b> tiene un ``.next_sibling``, pero no ``.previous_sibling``,
porque no hay nada antes de la etiqueta <b> `al mismo nivel del árbol`.
Por la misma razón, la etiqueta <c> tiene un ``.previous_sibling`` pero no
un ``.next_sibling``::

 print(sibling_soup.b.previous_sibling)
 # None
 print(sibling_soup.c.next_sibling)
 # None

Las cadenas "text1" y "text2" `no` son hermanas, porque no tienen la misma
madre::

 sibling_soup.b.string
 # 'text1'

 print(sibling_soup.b.string.next_sibling)
 # None

En documentos reales, los ``.next_sibling`` o ``.previous_sibling`` de
una etiqueta normalmente serán cadenas que contengan espacios en blanco.
Retomando el documento de "Las tres hermanas"::

 # <a href="http://example.com/elsie" class="sister" id="link1">Elsie</a>,
 # <a href="http://example.com/lacie" class="sister" id="link2">Lacie</a> and
 # <a href="http://example.com/tillie" class="sister" id="link3">Tillie</a>;

Podrías pensar que la ``.next_sibling`` de la primera etiqueta <a> podría
ser la segunda etiqueta <a>. Pero realmente es una cadena de caracteres:
la coma y el salto de línea que separan la primera etiqueta <a> de la
segunda::

 link = soup.a
 link
 # <a class="sister" href="http://example.com/elsie" id="link1">Elsie</a>

 link.next_sibling
 # ',\n '

La segunda etiqueta <a> es realmente la ``.next_sibling`` de la coma::

 link.next_sibling.next_sibling
 # <a class="sister" href="http://example.com/lacie" id="link2">Lacie</a>

.. _sibling-generators:

``.next_siblings`` y ``.previous_siblings``
-------------------------------------------

Puedes iterar sobre las hermanas de una etiqueta con ``.next_siblings`` o
``.previuos_siblings``::

 for sibling in soup.a.next_siblings:
     print(repr(sibling))
 # ',\n'
 # <a class="sister" href="http://example.com/lacie" id="link2">Lacie</a>
 # ' and\n'
 # <a class="sister" href="http://example.com/tillie" id="link3">Tillie</a>
 # '; and they lived at the bottom of a well.'

 for sibling in soup.find(id="link3").previous_siblings:
     print(repr(sibling))
 # ' and\n'
 # <a class="sister" href="http://example.com/lacie" id="link2">Lacie</a>
 # ',\n'
 # <a class="sister" href="http://example.com/elsie" id="link1">Elsie</a>
 # 'Once upon a time there were three little sisters; and their names were\n'

Hacia delante y hacia atrás
===========================

Échale un vistazo al comienzo del documento de "Las tres hermanas"::

 # <html><head><title>The Dormouse's story</title></head>
 # <p class="title"><b>The Dormouse's story</b></p>

Un analizador HTML toma esta cadena de caracteres y la convierte en
una serie de eventos: "se abre una etiqueta <html>", "se abre una
etiqueta <head>", "se abre una etiqueta <title>", "se añade una cadena",
"se cierra la etiqueta <title>", "se abre una etiqueta <p>" y así
sucesivamente. Beautiful Soup ofrece herramientas para reconstruir
el análisis inicial del documento.

.. _element-generators:

``.next_element`` y ``.previous_element``
-----------------------------------------

El atributo ``.next_element`` de una cadena o etiqueta apunta a cualquiera
que fue analizado inmediatamente después. Podría ser igual que ``.next_sibling``,
pero normalmente es drásticamente diferente.

Aquí está la etiqueta final <a> en el documento de "Las tres hermanas".
Su ``..next_sibling`` es una cadena: la terminación de la oración fue
interrumpida por el comienzo de la etiqueta <a>.::

 last_a_tag = soup.find("a", id="link3")
 last_a_tag
 # <a class="sister" href="http://example.com/tillie" id="link3">Tillie</a>

 last_a_tag.next_sibling
 # ';\nand they lived at the bottom of a well.'

Pero el ``.next_element`` de esa etiqueta <a>, lo que fue analizado
inmediatamente después de la etiqueta <a>, `no` es el resto de la
oración: es la palabra "Tillie"::

 last_a_tag.next_element
 # 'Tillie'

Esto se debe a que en el marcado original, la palabra "Tillie"
aparece antes del punto y coma. El analizador se encontró con
una etiqueta <a>, después la palabra "Tillie", entonces la etiqueta
de cierre </a>, después el punto y coma y el resto de la oración.
El punto y coma está al mismo nivel que la etiqueta <a>, pero
la palabra "Tillie" se encontró primera.

El atributo ``.previous_element`` es exactamente el opuesto
de ``.next_element``. Apunta a cualquier elemento que
fue analizado inmediatamente antes que este::

 last_a_tag.previous_element
 # ' and\n'
 last_a_tag.previous_element.next_element
 # <a class="sister" href="http://example.com/tillie" id="link3">Tillie</a>

``.next_elements`` y ``.previous_elements``
-------------------------------------------

Ya te estarás haciendo a la idea. Puedes usar estos iteradores
para moverte hacia delante y hacia atrás en el documento tal y como
fue analizado::

 for element in last_a_tag.next_elements:
     print(repr(element))
 # 'Tillie'
 # ';\nand they lived at the bottom of a well.'
 # '\n'
 # <p class="story">...</p>
 # '...'
 # '\n'

======================
 Buscar en el árbol
======================

Beautiful Soup define una gran cantidad de métodos para buscar en
el árbol analizado, pero todos son muy similares. Dedicaré mucho
tiempo explicando los dos métodos más populares: ``find()`` y
``find_all()``. Los otros métodos toman casi los mismos argumentos,
así que los cubriré brevemente.

De nuevo, usaré el documento de "Las tres hermanas" como ejemplo::

 html_doc = """
 <html><head><title>The Dormouse's story</title></head>
 <body>
 <p class="title"><b>The Dormouse's story</b></p>

 <p class="story">Once upon a time there were three little sisters; and their names were
 <a href="http://example.com/elsie" class="sister" id="link1">Elsie</a>,
 <a href="http://example.com/lacie" class="sister" id="link2">Lacie</a> and
 <a href="http://example.com/tillie" class="sister" id="link3">Tillie</a>;
 and they lived at the bottom of a well.</p>

 <p class="story">...</p>
 """

 from bs4 import BeautifulSoup
 soup = BeautifulSoup(html_doc, 'html.parser')

Empleando en un filtro un argumento como ``find_all()``, puedes
"acercar" aquellas partes del documento en las que estés interesado.

Tipos de filtros
================

Antes de entrar en detalle sobre ``find_all()`` y métodos similares,
me gustaría mostrar ejemplos de diferentes filtros que puedes
utilizar en estos métodos. Estos filtros aparecen una y otra vez a lo
largo de la API. Puedes usarlos para filtrar basándote en el nombre de
una etiqueta, en sus atributos, en el texto de una cadena, o en alguna
combinación de estos.

.. _a string:

Una cadena
----------

El filtro más simple es una cadena. Pasa una cadena a un método de
búsqueda y Beautiful Soup buscará un resultado para esa cadena
exactamente. Este código encuentra todas las etiquetas <b> en el
documento::

 soup.find_all('b')
 # [<b>The Dormouse's story</b>]

Si pasas un cadena de *bytes*, Beautiful Soup asumirá que la cadena
está codificada como UTF-8. Puedes evitar esto pasando una cadena
Unicode.

.. _a regular expression:

Una expresión regular
---------------------

Si pasas un objeto que sea una expresión regular, Beautiful Soup filtrará
mediante dicho expresión regular usando si su método ``search()``. Este
código encuentra todas las etiquetas cuyo nombre empiece por la letra
"b"; en este caso, las etiquetas <body> y <b>::

 import re
 for tag in soup.find_all(re.compile("^b")):
     print(tag.name)
 # body
 # b

Este código encuentra todas las etiquetas cuyo nombre contiene
la letra 't'::

 for tag in soup.find_all(re.compile("t")):
     print(tag.name)
 # html
 # title

.. _a list:

Una lista
---------

Si pasas una lista, Beautiful Soup hará una búsqueda por cadenas
con `cualquier` elemento en dicha lista. Este código encuentra
todas las etiquetas <a> `y` todas las etiquetas <b>::

 soup.find_all(["a", "b"])
 # [<b>The Dormouse's story</b>,
 #  <a class="sister" href="http://example.com/elsie" id="link1">Elsie</a>,
 #  <a class="sister" href="http://example.com/lacie" id="link2">Lacie</a>,
 #  <a class="sister" href="http://example.com/tillie" id="link3">Tillie</a>]

.. _the value True:

``True``
--------

El valor ``True`` empareja todo lo que pueda. Este código encuentra
``todas`` las etiquetas del documento, pero ninguna de las cadenas
de texto::

 for tag in soup.find_all(True):
     print(tag.name)
 # html
 # head
 # title
 # body
 # p
 # b
 # p
 # a
 # a
 # a
 # p

.. a function:

Una función
-----------

Si ninguna de las formas de búsqueda anteriores te sirven, define
una función que tome un elemento como su único argumento. La función
debería devolver ``True`` si el argumento se corresponde con lo indicado
en la función, y ``Falso`` en cualquier otro caso.

Esta es una función que devuelve ``True`` si una etiqueta tiene
definida el atributo "class" pero no el atributo "id"::

 def has_class_but_no_id(tag):
     return tag.has_attr('class') and not tag.has_attr('id')

Pasa esta función a ``find_all()`` y obtendrás todas las etiquetas
<p>::

 soup.find_all(has_class_but_no_id)
 # [<p class="title"><b>The Dormouse's story</b></p>,
 #  <p class="story">Once upon a time there were…bottom of a well.</p>,
 #  <p class="story">...</p>]

Esta función solo devuelve las etiquetas <p>. No obtiene las etiquetas
<a>, porque esas etiquetas definen ambas "class" y "id". No devuelve
etiquetas como <html> y <title> porque dichas etiquetas no definen
"class".

Si pasas una función para filtrar un atributo en específico como
``href``, el argumento que se pasa a la función será el valor de
dicho atributo, no toda la etiqueta. Esta es una función que
encuentra todas las etiquetas <a> cuyo atributo ``href`` *no*
empareja con una expresión regular::

 import re
 def not_lacie(href):
     return href and not re.compile("lacie").search(href)
 
 soup.find_all(href=not_lacie)
 # [<a class="sister" href="http://example.com/elsie" id="link1">Elsie</a>,
 #  <a class="sister" href="http://example.com/tillie" id="link3">Tillie</a>]

La función puede ser tan complicada como la necesites. Esta es una
función que devuelve ``True`` si una etiqueta está rodeada por
objetos *string*::

 from bs4 import NavigableString
 def surrounded_by_strings(tag):
     return (isinstance(tag.next_element, NavigableString)
             and isinstance(tag.previous_element, NavigableString))

 for tag in soup.find_all(surrounded_by_strings):
     print(tag.name)
 # body
 # p
 # a
 # a
 # a
 # p

Ahora ya estamos listos para entrar en detalle en los métodos
de búsqueda.

``find_all()``
==============

Firma del método: find_all(:ref:`name <name>`, :ref:`attrs <attrs>`, :ref:`recursive
<recursive>`, :ref:`string <string>`, :ref:`limit <limit>`, :ref:`**kwargs <kwargs>`)

El método ``find_all()`` busca por los descendientes de una etiqueta y
obtiene `todos` aquellos que casan con tus filtros. He mostrado varios
ejemplos en `Tipos de filtros`_, pero aquí hay unos cuantos más::

 soup.find_all("title")
 # [<title>The Dormouse's story</title>]

 soup.find_all("p", "title")
 # [<p class="title"><b>The Dormouse's story</b></p>]

 soup.find_all("a")
 # [<a class="sister" href="http://example.com/elsie" id="link1">Elsie</a>,
 #  <a class="sister" href="http://example.com/lacie" id="link2">Lacie</a>,
 #  <a class="sister" href="http://example.com/tillie" id="link3">Tillie</a>]

 soup.find_all(id="link2")
 # [<a class="sister" href="http://example.com/lacie" id="link2">Lacie</a>]

 import re
 soup.find(string=re.compile("sisters"))
 # 'Once upon a time there were three little sisters; and their names were\n'

Algunos de estos deberían ser familiares, pero otros son nuevos.
¿Qué significa pasar un valor para ``string``, o ``id``? ¿Por qué
``find_all("p", "title")`` encuentra una etiqueta <p> con la clase
CSS "title"? Echemos un vistazo a los argumentos de ``find_all()``.

.. _name:

El argumento ``name``
---------------------

Pasa un valor para ``name`` y notarás que Beautiful Soup solo
considera etiquetas con ciertos nombres. Las cadenas de texto se
ignorarán, como aquellas etiquetas cuyo nombre no emparejen.

Este es el uso más simple::

 soup.find_all("title")
 # [<title>The Dormouse's story</title>]

Recuerda de `Tipos de filtros`_ que el valor para ``name`` puede ser
`una cadena`_, `una expresión regular`_, `una lista`_, `una función`_,
o el valor `True`_.

.. _kwargs:

El argumento palabras-clave
---------------------------

Cualquier argumento que no se reconozca se tomará como un filtro para alguno
de los atributos de una etiqueta. Si pasas un valor para un argumento llamado
``id``, Beautiful Soup filtrará el atributo 'id' de cada una de las etiquetas::

 soup.find_all(id='link2')
 # [<a class="sister" href="http://example.com/lacie" id="link2">Lacie</a>]

Si pasas un valor para ``href``, Beautiful Soup filtrará
el atributo ``href`` de cada uno de las etiquetas::

 soup.find_all(href=re.compile("elsie"))
 # [<a class="sister" href="http://example.com/elsie" id="link1">Elsie</a>]

Puedes filtrar un atributo basándote en `una cadena`_,
`una expresión regular`_, `una lista`_, `una función`_, o el valor
`True`_.

Este código busca todas las etiquetas cuyo atributo ``id`` tiene
un valor, sin importar qué valor es::

 soup.find_all(id=True)
 # [<a class="sister" href="http://example.com/elsie" id="link1">Elsie</a>,
 #  <a class="sister" href="http://example.com/lacie" id="link2">Lacie</a>,
 #  <a class="sister" href="http://example.com/tillie" id="link3">Tillie</a>]

Puedes filtrar varios atributos al mismo tiempo pasando más de un argumento
palabra-clave::

 soup.find_all(href=re.compile("elsie"), id='link1')
 # [<a class="sister" href="http://example.com/elsie" id="link1">Elsie</a>]

Algunos atributos, como los atributos data-* en HTML5, tienen nombres que
no pueden ser usados como nombres de argumentos palabra-clave::

 data_soup = BeautifulSoup('<div data-foo="value">foo!</div>', 'html.parser')
 data_soup.find_all(data-foo="value")
 # SyntaxError: keyword can't be an expression

Puedes usar estos atributos en búsquedas insertándolos en un diccionario
y pasándolo a ``find_all()`` como el argumento ``attrs``::

 data_soup.find_all(attrs={"data-foo": "value"})
 # [<div data-foo="value">foo!</div>]

No puedes usar un argumento palabra-clave para buscar por el nombre
HTML de un elemento, porque BeautifulSoup usa el argumento ``name``
para guardar el nombre de la etiqueta. En lugar de esto, puedes
darle valor a 'name' en el argumento ``attrs``::

 name_soup = BeautifulSoup('<input name="email"/>', 'html.parser')
 name_soup.find_all(name="email")
 # []
 name_soup.find_all(attrs={"name": "email"})
 # [<input name="email"/>]

.. _attrs:

Buscando por clase CSS
----------------------

Es muy útil para buscar una etiqueta que tenga una clase CSS específica,
pero el nombre del atributo CSS, "class", es una palabra reservada de
Python. Usar ``class`` como argumento ocasionaría un error sintáctico.
Desde Beautiful Soup 4.1.2, se puede buscar por una clase CSS usando
el argumento palabra-clave ``class_``::

 soup.find_all("a", class_="sister")
 # [<a class="sister" href="http://example.com/elsie" id="link1">Elsie</a>,
 #  <a class="sister" href="http://example.com/lacie" id="link2">Lacie</a>,
 #  <a class="sister" href="http://example.com/tillie" id="link3">Tillie</a>]

Como con cualquier argumento palabra-clave, puede pasar una cadena
de caracteres a ``class_``, una expresión regular, una función, o
``True``::

 soup.find_all(class_=re.compile("itl"))
 # [<p class="title"><b>The Dormouse's story</b></p>]

 def has_six_characters(css_class):
     return css_class is not None and len(css_class) == 6

 soup.find_all(class_=has_six_characters)
 # [<a class="sister" href="http://example.com/elsie" id="link1">Elsie</a>,
 #  <a class="sister" href="http://example.com/lacie" id="link2">Lacie</a>,
 #  <a class="sister" href="http://example.com/tillie" id="link3">Tillie</a>]

:ref:`Recuerda <multivalue>` que una sola etiqueta puede tener varios
valores para su atributo "class". Cuando se busca por una etiqueta
que case una cierta clase CSS, se está intentando emparejar por
`cualquiera` de sus clases CSS::

 css_soup = BeautifulSoup('<p class="body strikeout"></p>', 'html.parser')
 css_soup.find_all("p", class_="strikeout")
 # [<p class="body strikeout"></p>]

 css_soup.find_all("p", class_="body")
 # [<p class="body strikeout"></p>]

Puedes también buscar por la cadena de caracteres exacta del atributo
``class``::

 css_soup.find_all("p", class_="body strikeout")
 # [<p class="body strikeout"></p>]

Pero buscar por variantes de la cadena de caracteres no funcionará::

 css_soup.find_all("p", class_="strikeout body")
 # []

Si quieres buscar por las etiquetas que casen dos o más clases CSS,
deberías usar un selector CSS::

 css_soup.select("p.strikeout.body")
 # [<p class="body strikeout"></p>]

En versiones antiguas de Beautiful Soup, que no soportan el
atajo ``class_``, puedes usar el truco del ``attrs`` mencionado
arriba. Crea un diccionario cuyo valor para "class" sea la
cadena de caracteres (o expresión regular, o lo que sea) que
quieras buscar::

 soup.find_all("a", attrs={"class": "sister"})
 # [<a class="sister" href="http://example.com/elsie" id="link1">Elsie</a>,
 #  <a class="sister" href="http://example.com/lacie" id="link2">Lacie</a>,
 #  <a class="sister" href="http://example.com/tillie" id="link3">Tillie</a>]

.. _string:

El argumento ``string``
-----------------------

Con ``string`` puedes buscar por cadenas de caracteres en vez de
etiquetas. Como con ``name`` y argumentos palabras-clave, puedes
pasar `una cadena`_, `una expresión regular`_, `una lista`_, `una
función`_, o el valor `True`_.
Aquí hay algunos ejemplos::

 soup.find_all(string="Elsie")
 # ['Elsie']

 soup.find_all(string=["Tillie", "Elsie", "Lacie"])
 # ['Elsie', 'Lacie', 'Tillie']

 soup.find_all(string=re.compile("Dormouse"))
 # ["The Dormouse's story", "The Dormouse's story"]

 def is_the_only_string_within_a_tag(s):
     """Return True if this string is the only child of its parent tag."""
     return (s == s.parent.string)

 soup.find_all(string=is_the_only_string_within_a_tag)
 # ["The Dormouse's story", "The Dormouse's story", 'Elsie', 'Lacie', 'Tillie', '...']


Aunque ``string`` es para encontrar cadenas, puedes combinarlo
con argumentos que permitan buscar etiquetas: Beautiful Soup
encontrará todas las etiquetas cuyo ``.string`` case con tu valor
para ``string``. Este código encuentra las etiquetas <a> cuyo
``.string`` es "Elsie"::

 soup.find_all("a", string="Elsie")
 # [<a href="http://example.com/elsie" class="sister" id="link1">Elsie</a>]

El argumento ``string`` es nuevo en Beautiful Soup 4.4.0. En versiones
anteriores se llamaba ``text``::

 soup.find_all("a", text="Elsie")
 # [<a href="http://example.com/elsie" class="sister" id="link1">Elsie</a>]

.. _limit:

El argumento``limit``
---------------------

``find_all()`` devuelve todas las etiquetas y cadenas que emparejan
con tus filtros. Esto puede tardar un poco si el documento es grande.
Si no necesitas `todos` los resultados, puedes pasar un número para
``limit``. Esto funciona tal y como lo hace la palabra LIMIT en SQL.
Indica a Beautiful Soup dejar de obtener resultados después de
haber encontrado un cierto número.

Hay tres enlaces en el documento de "Las tres hermanas", pero este
código tan solo obtiene los dos primeros::

 soup.find_all("a", limit=2)
 # [<a class="sister" href="http://example.com/elsie" id="link1">Elsie</a>,
 #  <a class="sister" href="http://example.com/lacie" id="link2">Lacie</a>]

.. _recursive:

El argumento ``recursive``
--------------------------

Si llamas a ``mytag.find_all()``, Beautiful Soup examinará todos los
descendientes de ``mytag``: sus hijos, los hijos de sus hijos, y
así sucesivamente. Si solo quieres que Beautiful Soup considere
hijos directos, puedes pasar ``recursive=False``. Observa las
diferencias aquí::

 soup.html.find_all("title")
 # [<title>The Dormouse's story</title>]

 soup.html.find_all("title", recursive=False)
 # []

Aquí está esa parte del documento::

 <html>
  <head>
   <title>
    The Dormouse's story
   </title>
  </head>
 ...

La etiqueta <title> va después de la etiqueta <html>, pero no está
`directamente` debajo de la etiqueta <html>: la etiqueta <head>
está en medio de ambas. Beautiful Soup encuentra la etiqueta <title> cuando
se permite observar todos los descendientes de la etiqueta <html>,
pero cuando ``recursive=False`` restringe a los hijos directos
de la etiqueta <html>, no se encuentra nada.

Beautiful Soup ofrece mucho métodos de análisis del árbol (descritos
más adelante), y la mayoría toman los mismos argumentos que ``find_all()``:
``name``, ``attrs``, ``string``, ``limit``, y los argumentos
palabras-clave. Pero el argumento ``recursive`` es diferente:
``find_all()`` y ``find()`` son los únicos métodos que lo soportan.
Pasar ``recursive=False`` en un método como ``find_parents()`` no sería
muy útil.

Llamar a una etiqueta es como llamar a ``find_all()``
=====================================================

Como ``find_all()`` es el método más popular en la API de búsqueda
de Beautiful Soup, puedes usar un atajo para usarlo. Si utilizas
el objeto :py:class:`BeautifulSoup` o un objeto :py:class:`Tag`
como si fuesen una función, entonces es lo mismo que llamar a
``find_all()`` en esos objetos. Estos dos líneas de código son
equivalentes::

 soup.find_all("a")
 soup("a")

Estas dos líneas de código son también equivalentes::

 soup.title.find_all(string=True)
 soup.title(string=True)

``find()``
==========

Firma del método: find(:ref:`name <name>`, :ref:`attrs <attrs>`, :ref:`recursive
<recursive>`, :ref:`string <string>`, :ref:`**kwargs <kwargs>`)

El método ``find_all()`` examina todo el documento buscando por
resultados, pero a veces solo quieres encontrar un resultado.
Si sabes que un documento solo tiene una etiqueta <body>, es una
pérdida de tiempo examinar todo el documento buscando más
emparejamientos. En lugar de pasar ``limit=1`` siempre que se llame
a ``find_all(), puedes usar el método ``find()``. Estas dos líneas
de código son `casi` equivalentes::

 soup.find_all('title', limit=1)
 # [<title>The Dormouse's story</title>]

 soup.find('title')
 # <title>The Dormouse's story</title>

La única diferencia es que ``find_all()`` devuelve una lista
conteniendo un resultado, y ``find()`` devuelve solo el resultado.

Si ``find_all()`` no encuentra nada, devuelve una lista vacía. Si
``find()`` no encuentra nada, devuelve ``None``::

 print(soup.find("nosuchtag"))
 # None

¿Recuerdas el truco de ``soup.head.title`` de `Navegar usando nombres
de etiquetas`_? Ese truco funciona porque se llama repetidamente a
``find()``::

 soup.head.title
 # <title>The Dormouse's story</title>

 soup.find("head").find("title")
 # <title>The Dormouse's story</title>

``find_parents()`` y ``find_parent()``
======================================

Firma del método: find_parents(:ref:`name <name>`, :ref:`attrs <attrs>`, :ref:`string <string>`, :ref:`limit <limit>`, :ref:`**kwargs <kwargs>`)

Firma del método: find_parent(:ref:`name <name>`, :ref:`attrs <attrs>`, :ref:`string <string>`, :ref:`**kwargs <kwargs>`)

He pasado bastante tiempo cubriendo ``find_all()`` y ``find()``.
La API de Beautiful Soup define otros diez métodos para buscar por
el árbol, pero no te asustes. Cinco de estos métodos son básicamente
iguales a ``find_all()``, y los otros cinco son básicamente
iguales a ``find()``. La única diferencia reside en qué partes del
árbol buscan.

Primero consideremos ``find_parents()`` y ``find_paren()``. Recuerda
que ``find_all()`` y ``find()`` trabajan bajando por el árbol,
examinando a los descendientes de una etiqueta. Estos métodos realizan
lo contrario: trabajan `subiendo` por el árbol, buscando a las madres
de las etiquetas (o cadenas). Probémoslos, empezando por una cadena
de caracteres que esté bien enterrada en el documento de "Las tres
hermanas"::

 a_string = soup.find(string="Lacie")
 a_string
 # 'Lacie'

 a_string.find_parents("a")
 # [<a class="sister" href="http://example.com/lacie" id="link2">Lacie</a>]

 a_string.find_parent("p")
 # <p class="story">Once upon a time there were three little sisters; and their names were
 #  <a class="sister" href="http://example.com/elsie" id="link1">Elsie</a>,
 #  <a class="sister" href="http://example.com/lacie" id="link2">Lacie</a> and
 #  <a class="sister" href="http://example.com/tillie" id="link3">Tillie</a>;
 #  and they lived at the bottom of a well.</p>

 a_string.find_parents("p", class_="title")
 # []

Una de la tres etiquetas <a> is la madre directa de la cadena
en cuestión, así que nuestra búsqueda la encuentra. Una de las
tres etiquetas <p> es una madre indirecta de la cadena, y nuestra
búsqueda también la encuentra. Hay una etiqueta <p> con la clase
CSS "title" `en algún sitio` del documento, pero no en ninguno
de las madres de la cadena, así que no podemos encontrarla con
``find_parents()``.

Puedes haber deducido la conexión entre ``find_parent()`` y
``find_parents()``, y los atributos `.parent`_ y `.parents`_
mencionados anteriormente. La conexión es muy fuerte. Estos
métodos de búsqueda realmente usan ``.parents`` para iterar
sobre todas las madres, y comprobar cada una con el filtro
provisto para ver si emparejan.

``find_next_siblings()`` y ``find_next_sibling()``
==================================================

Firma del método: find_next_siblings(:ref:`name <name>`, :ref:`attrs <attrs>`, :ref:`string <string>`, :ref:`limit <limit>`, :ref:`**kwargs <kwargs>`)

Firma del método: find_next_sibling(:ref:`name <name>`, :ref:`attrs <attrs>`, :ref:`string <string>`, :ref:`**kwargs <kwargs>`)

Estos métodos usan :ref:`next_siblings <sibling-generators>`
para iterar sobre el resto de los hermanos de un elemento en el
árbol. El método ``find_next_siblings()`` devuelve todos los
hermanos que casen, y ``find_next_sibling()`` solo devuelve
el primero de ellos::

 first_link = soup.a
 first_link
 # <a class="sister" href="http://example.com/elsie" id="link1">Elsie</a>

 first_link.find_next_siblings("a")
 # [<a class="sister" href="http://example.com/lacie" id="link2">Lacie</a>,
 #  <a class="sister" href="http://example.com/tillie" id="link3">Tillie</a>]

 first_story_paragraph = soup.find("p", "story")
 first_story_paragraph.find_next_sibling("p")
 # <p class="story">...</p>

``find_previous_siblings()`` y ``find_previous_sibling()``
==========================================================

Firma del método: find_previous_siblings(:ref:`name <name>`, :ref:`attrs <attrs>`, :ref:`string <string>`, :ref:`limit <limit>`, :ref:`**kwargs <kwargs>`)

Firma del método: find_previous_sibling(:ref:`name <name>`, :ref:`attrs <attrs>`, :ref:`string <string>`, :ref:`**kwargs <kwargs>`)

Estos métodos emplean :ref:`.previous_siblings <sibling-generators>` para iterar sobre
los hermanos de un elemento que les precede en el árbol. El método
``find_previous_siblings()`` devuelve todos los hermanos que emparejan, y
``find_previous_sibling()`` solo devuelve el primero de ellos::

 last_link = soup.find("a", id="link3")
 last_link
 # <a class="sister" href="http://example.com/tillie" id="link3">Tillie</a>

 last_link.find_previous_siblings("a")
 # [<a class="sister" href="http://example.com/lacie" id="link2">Lacie</a>,
 #  <a class="sister" href="http://example.com/elsie" id="link1">Elsie</a>]

 first_story_paragraph = soup.find("p", "story")
 first_story_paragraph.find_previous_sibling("p")
 # <p class="title"><b>The Dormouse's story</b></p>


``find_all_next()`` y ``find_next()``
=====================================

Firma del método: find_all_next(:ref:`name <name>`, :ref:`attrs <attrs>`, :ref:`string <string>`, :ref:`limit <limit>`, :ref:`**kwargs <kwargs>`)

Firma del método: find_next(:ref:`name <name>`, :ref:`attrs <attrs>`, :ref:`string <string>`, :ref:`**kwargs <kwargs>`)

Estos métodos usan :ref:`.next_elements <element-generators>` para
iterar sobre cualesquiera etiquetas y cadenas que vayan después
de ella en el documento. El método ``find_all_next()`` devuelve
todos los resultados, y ``find_next()`` solo devuelve el primero::

 first_link = soup.a
 first_link
 # <a class="sister" href="http://example.com/elsie" id="link1">Elsie</a>

 first_link.find_all_next(string=True)
 # ['Elsie', ',\n', 'Lacie', ' and\n', 'Tillie',
 #  ';\nand they lived at the bottom of a well.', '\n', '...', '\n']

 first_link.find_next("p")
 # <p class="story">...</p>

En el primer ejemplo, la cadena "Elsie" apareció, aunque estuviese
contenida en la etiqueta <a> desde la que comenzamos. En el segundo
ejemplo, la última etiqueta <p> en el documento apareció, aunque no
esté en la misma parte del árbol que la etiqueta <a> desde la que
comenzamos. Para estos métodos, todo lo que importa es que un
elemento cumple con el filtro, y que aparezca en el documento
después del elemento inicial.

``find_all_previous()`` y ``find_previous()``
=============================================

Firma del método: find_all_previous(:ref:`name <name>`, :ref:`attrs <attrs>`, :ref:`string <string>`, :ref:`limit <limit>`, :ref:`**kwargs <kwargs>`)

Firma del método: find_previous(:ref:`name <name>`, :ref:`attrs <attrs>`, :ref:`string <string>`, :ref:`**kwargs <kwargs>`)

Estos métodos usan :ref:`.previous_elements <element-generators>`
para iterar sobre las etiquetas y cadenas que iban antes en el
documento. El método ``find_all_previous()`` devuelve todos los
resultados, y ``find_previous()`` solo devuelve el primero::

 first_link = soup.a
 first_link
 # <a class="sister" href="http://example.com/elsie" id="link1">Elsie</a>

 first_link.find_all_previous("p")
 # [<p class="story">Once upon a time there were three little sisters; ...</p>,
 #  <p class="title"><b>The Dormouse's story</b></p>]

 first_link.find_previous("title")
 # <title>The Dormouse's story</title>

La llamada a ``find_all_previous("p")`` encontró el primer
párrafo en el documento (el que tiene la clase="title"), pero
también encuentra el segundo párrafo, la etiqueta <p> que
contiene la etiqueta <a> con la que comenzamos. Esto no debería
ser demasiado sorprendente: estamos buscando todas las etiquetas
que aparecen en el documento después de la etiqueta con la que se
comienza. Una etiqueta <p> que contiene una <a> debe aparecer
antes de la etiqueta <a> que contiene.

Selectores CSS mediante la propiedad ``.css``
=============================================

Los objetos :py:class:`BeautifulSoup` y :py:class:`Tag` soportan los selectores
CSS a través de su atributo ``.css``. El paquete `Soup Sieve <https://facelessuser.github.io/soupsieve/>`_,
disponible a través de PyPI como ``soupsieve``, gestiona la implementación real
del selector. Si instalaste Beautiful Soup mediante ``pip``, Soup Sieve se
instaló al mismo tiempo, así que no tienes que hacer nada adicional.

La documentación de Soup Sieve lista `todos los selectores CSS soportados
actualmente <https://facelessuser.github.io/soupsieve/selectors/>`_, pero
estos son algunos de los básicos. Puedes encontrar etiquetas::

 soup.css.select("title")
 # [<title>The Dormouse's story</title>]

 soup.css.select("p:nth-of-type(3)")
 # [<p class="story">...</p>]

Encontrar etiquetas dentro de otras etiquetas::

 soup.css.select("body a")
 # [<a class="sister" href="http://example.com/elsie" id="link1">Elsie</a>,
 #  <a class="sister" href="http://example.com/lacie"  id="link2">Lacie</a>,
 #  <a class="sister" href="http://example.com/tillie" id="link3">Tillie</a>]

 soup.css.select("html head title")
 # [<title>The Dormouse's story</title>]

Encontrar etiquetas `directamente` después de otras etiquetas::

 soup.css.select("head > title")
 # [<title>The Dormouse's story</title>]

 soup.css.select("p > a")
 # [<a class="sister" href="http://example.com/elsie" id="link1">Elsie</a>,
 #  <a class="sister" href="http://example.com/lacie"  id="link2">Lacie</a>,
 #  <a class="sister" href="http://example.com/tillie" id="link3">Tillie</a>]

 soup.css.select("p > a:nth-of-type(2)")
 # [<a class="sister" href="http://example.com/lacie" id="link2">Lacie</a>]

 soup.css.select("p > #link1")
 # [<a class="sister" href="http://example.com/elsie" id="link1">Elsie</a>]

 soup.css.select("body > a")
 # []

Encontrar los hijos de etiquetas::

 soup.css.select("#link1 ~ .sister")
 # [<a class="sister" href="http://example.com/lacie" id="link2">Lacie</a>,
 #  <a class="sister" href="http://example.com/tillie"  id="link3">Tillie</a>]

 soup.css.select("#link1 + .sister")
 # [<a class="sister" href="http://example.com/lacie" id="link2">Lacie</a>]

Encontrar etiquetas por su clase CSS::

 soup.css.select(".sister")
 # [<a class="sister" href="http://example.com/elsie" id="link1">Elsie</a>,
 #  <a class="sister" href="http://example.com/lacie" id="link2">Lacie</a>,
 #  <a class="sister" href="http://example.com/tillie" id="link3">Tillie</a>]

 soup.css.select("[class~=sister]")
 # [<a class="sister" href="http://example.com/elsie" id="link1">Elsie</a>,
 #  <a class="sister" href="http://example.com/lacie" id="link2">Lacie</a>,
 #  <a class="sister" href="http://example.com/tillie" id="link3">Tillie</a>]

Encontrar etiquetas por su ID::

 soup.css.select("#link1")
 # [<a class="sister" href="http://example.com/elsie" id="link1">Elsie</a>]

 soup.css.select("a#link2")
 # [<a class="sister" href="http://example.com/lacie" id="link2">Lacie</a>]

Encontrar etiquetas que casen con cualquier selector que estés en una
lista de selectores::

 soup.css.select("#link1,#link2")
 # [<a class="sister" href="http://example.com/elsie" id="link1">Elsie</a>,
 #  <a class="sister" href="http://example.com/lacie" id="link2">Lacie</a>]

Comprobar la existencia de un atributo::

 soup.css.select('a[href]')
 # [<a class="sister" href="http://example.com/elsie" id="link1">Elsie</a>,
 #  <a class="sister" href="http://example.com/lacie" id="link2">Lacie</a>,
 #  <a class="sister" href="http://example.com/tillie" id="link3">Tillie</a>]

Encontrar etiquetas por el valor de un atributo::

 soup.css.select('a[href="http://example.com/elsie"]')
 # [<a class="sister" href="http://example.com/elsie" id="link1">Elsie</a>]

 soup.css.select('a[href^="http://example.com/"]')
 # [<a class="sister" href="http://example.com/elsie" id="link1">Elsie</a>,
 #  <a class="sister" href="http://example.com/lacie" id="link2">Lacie</a>,
 #  <a class="sister" href="http://example.com/tillie" id="link3">Tillie</a>]

 soup.css.select('a[href$="tillie"]')
 # [<a class="sister" href="http://example.com/tillie" id="link3">Tillie</a>]

 soup.css.select('a[href*=".com/el"]')
 # [<a class="sister" href="http://example.com/elsie" id="link1">Elsie</a>]

Hay también un método llamado ``select_one()``, que encuentra solo
la primera etiqueta que case con un selector::

 soup.css.select_one(".sister")
 # <a class="sister" href="http://example.com/elsie" id="link1">Elsie</a>

Por conveniencia, puedes llamar a ``select()`` y ``select_one()`` sobre
el objeto :py:class:`BeautifulSoup` o :py:class:`Tag`, omitiendo la
propiedad ``.css``::

 soup.select('a[href$="tillie"]')
 # [<a class="sister" href="http://example.com/tillie" id="link3">Tillie</a>]

 soup.select_one(".sister")
 # <a class="sister" href="http://example.com/elsie" id="link1">Elsie</a>

El soporte de selectores CSS es conveniente para personas que ya conocen
la sintaxis de los selectores CSS. Puedes hacer todo esto con la API
de Beautiful Soup. Si todo lo que necesitas son los selectores CSS, deberías
saltarte Beautiful Soup y analizar el documento con ``lxml``: es mucho más
rápido. Pero Soup Sieve te permite `combinar` selectores CSS con la API
de Beautiful Soup. 

Características avanzadas de Soup Sieve
---------------------------------------

Soup Sieve ofrece una API más amplia más allá de los métodos ``select()``
y ``select_one()``, y puedes acceder a casi toda esa API a través del
atributo ``.css`` de :py:class:`Tag` o :py:class:`Beautiful Soup`. Lo que
sigue es solo una lista de los métodos soportados; ve a `la documentación de
Soup Sieve <https://facelessuser.github.io/soupsieve/>`_ para la documentación
completa.

El método ``iselect()`` funciona igualmente que ``select()``, solo que
devuelve un generador en vez de una lista::

 [tag['id'] for tag in soup.css.iselect(".sister")]
 # ['link1', 'link2', 'link3']

El método ``closest()`` devuelve la madre más cercana de una :py:class:`Tag` dada
que case con un selector CSS, similar al método ``find_parent()`` de
Beautiful Soup::

 elsie = soup.css.select_one(".sister")
 elsie.css.closest("p.story")
 # <p class="story">Once upon a time there were three little sisters; and their names were
 #  <a class="sister" href="http://example.com/elsie" id="link1">Elsie</a>,
 #  <a class="sister" href="http://example.com/lacie" id="link2">Lacie</a> and
 #  <a class="sister" href="http://example.com/tillie" id="link3">Tillie</a>;
 #  and they lived at the bottom of a well.</p>

El método ``match()`` devuelve un booleano dependiendo de si
una :py:class:`Tag` específica casa con un selector o no::
 
 # elsie.css.match("#link1")
 True

 # elsie.css.match("#link2")
 False

El método ``filter()`` devuelve un subconjunto de los hijos directos
de una etiqueta que casen con un selector::
 
 [tag.string for tag in soup.find('p', 'story').css.filter('a')]
 # ['Elsie', 'Lacie', 'Tillie']

El método ``escape()`` formatea los identificadores CSS que de otra
forma serían inválidos::
 
 soup.css.escape("1-strange-identifier")
 # '\\31 -strange-identifier'

Espacios de nombres en selectores CSS
-------------------------------------
Si has analizado XML que define espacios de nombres, puedes usarlos
en selectores CSS::

 from bs4 import BeautifulSoup
 xml = """<tag xmlns:ns1="http://namespace1/" xmlns:ns2="http://namespace2/">
  <ns1:child>I'm in namespace 1</ns1:child>
  <ns2:child>I'm in namespace 2</ns2:child>
 </tag> """
 namespace_soup = BeautifulSoup(xml, "xml")

 namespace_soup.css.select("child")
 # [<ns1:child>I'm in namespace 1</ns1:child>, <ns2:child>I'm in namespace 2</ns2:child>]

 namespace_soup.css.select("ns1|child")
 # [<ns1:child>I'm in namespace 1</ns1:child>]

Beautiful Soup intenta usar prefijos de espacios de nombres que tengan
sentido basándose en lo que vio al analizar el documento, pero siempre
puedes indicar tu propio diccionario de abreviaciones::

 namespaces = dict(first="http://namespace1/", second="http://namespace2/")
 namespace_soup.css.select("second|child", namespaces=namespaces)
 # [<ns1:child>I'm in namespace 2</ns1:child>]

Historia del soporte de selectores CSS
--------------------------------------

La propiedad ``.css`` fue añadida en Beautiful Soup 4.12.0. Anterior a esta,
solo los métodos convenientes ``.select()`` y ``select_one()`` se
soportaban.

La integración de Soup Sieve fue añadida en Beautiful Soup 4.7.0. Versiones
anteriores tenían el método ``.select()``, pero solo los selectores CSS
más comunes eran admitidos.
 

====================
 Modificar el árbol
====================

La mayor fortaleza de Beautiful Soup reside en buscar en el árbol
analizado, pero puedes también modificar el árbol y escribir tus
cambios como un nuevo documento HTML o XML.

Cambiar nombres de etiquetas y atributos
========================================

Cubrí esto anteriormente, en :py:class:`Tag.attrs`, pero vale la pena
repetirlo. Puedes renombrar una etiqueta, cambiar el valor de sus
atributos, añadir nuevos atributos, y eliminar atributos::

 soup = BeautifulSoup('<b class="boldest">Extremely bold</b>', 'html.parser')
 tag = soup.b

 tag.name = "blockquote"
 tag['class'] = 'verybold'
 tag['id'] = 1
 tag
 # <blockquote class="verybold" id="1">Extremely bold</blockquote>

 del tag['class']
 del tag['id']
 tag
 # <blockquote>Extremely bold</blockquote>

Modificar ``.string``
=====================

Si quieres establecer el ``.string`` de una etiqueta a una nueva cadena de
caracteres, los contenidos de la etiqueta se pueden reemplazar con esa cadena::

 markup = '<a href="http://example.com/">I linked to <i>example.com</i></a>'
 soup = BeautifulSoup(markup, 'html.parser')

 tag = soup.a
 tag.string = "New link text."
 tag
 # <a href="http://example.com/">New link text.</a>

Ten cuidado: si una etiqueta contiene otras, ellas y todo su contenido
serán destruidos.  

``append()``
============

Puedes añadir al contenido de una etiqueta con ``Tag.append()``.
Funciona como llamar a ``.append()`` en una lista de Python::

 soup = BeautifulSoup("<a>Foo</a>", 'html.parser')
 soup.a.append("Bar")

 soup
 # <a>FooBar</a>
 soup.a.contents
 # ['Foo', 'Bar']

``extend()``
============

Desde Beautiful Soup 4.7.0, :py:class:`Tag` también soporta un método
llamado ``.extend()``, el cual añade todos los elementos de una lista
a una :py:class:`Tag`, en orden::

 soup = BeautifulSoup("<a>Soup</a>", 'html.parser')
 soup.a.extend(["'s", " ", "on"])

 soup
 # <a>Soup's on</a>
 soup.a.contents
 # ['Soup', ''s', ' ', 'on']
   
``NavigableString()`` y ``.new_tag()``
======================================

Si necesitas añadir una cadena a un documento, sin problema--puedes
pasar una cadena de Python a ``append()``, o puedes llamar al constructor
de :py:class:`NavigableString`::

 from bs4 import NavigableString
 soup = BeautifulSoup("<b></b>", 'html.parser')
 tag = soup.b
 tag.append("Hello")
 new_string = NavigableString(" there")
 tag.append(new_string)
 tag
 # <b>Hello there.</b>
 tag.contents
 # ['Hello', ' there']

Si quieres crear un comentario o cualquier otra subclase
de :py:class:`NavigableString`, solo llama al constructor::

 from bs4 import Comment
 new_comment = Comment("Nice to see you.")
 tag.append(new_comment)
 tag
 # <b>Hello there<!--Nice to see you.--></b>
 tag.contents
 # ['Hello', ' there', 'Nice to see you.']

`(Esto es una nueva característica en Beautiful Soup 4.4.0.)`

¿Qué ocurre si necesitas crear una etiqueta totalmente nueva? La mejor
solución es llamar al método de construcción (`factory method`)
``BeautifulSoup.new_tag()``::

 soup = BeautifulSoup("<b></b>", 'html.parser')
 original_tag = soup.b

 new_tag = soup.new_tag("a", href="http://www.example.com")
 original_tag.append(new_tag)
 original_tag
 # <b><a href="http://www.example.com"></a></b>

 new_tag.string = "Link text."
 original_tag
 # <b><a href="http://www.example.com">Link text.</a></b>

Solo el primer argumento, el nombre de la etiqueta, es
obligatorio.

``insert()``
============

``Tag.insert()`` es justo como ``Tag.append()``, excepto que el nuevo
elemento no necesariamente va al final del ``.contents`` de su madre.
Se insertará en la posición numérica que le hayas indicado. Funciona
como ``.insert()`` es una lista de Python::

 markup = '<a href="http://example.com/">I linked to <i>example.com</i></a>'
 soup = BeautifulSoup(markup, 'html.parser')
 tag = soup.a

 tag.insert(1, "but did not endorse ")
 tag
 # <a href="http://example.com/">I linked to but did not endorse <i>example.com</i></a>
 tag.contents
 # ['I linked to ', 'but did not endorse', <i>example.com</i>]

``insert_before()`` y ``insert_after()``
========================================

El método ``insert_before()`` inserta etiquetas o cadenas
inmediatamente antes de algo en el árbol analizado::

 soup = BeautifulSoup("<b>leave</b>", 'html.parser')
 tag = soup.new_tag("i")
 tag.string = "Don't"
 soup.b.string.insert_before(tag)
 soup.b
 # <b><i>Don't</i>leave</b>

El método ``insert_after()`` inserta etiquetas o cadenas
inmediatamente después de algo en el árbol analizado::

 div = soup.new_tag('div')
 div.string = 'ever'
 soup.b.i.insert_after(" you ", div)
 soup.b
 # <b><i>Don't</i> you <div>ever</div> leave</b>
 soup.b.contents
 # [<i>Don't</i>, ' you', <div>ever</div>, 'leave']

``clear()``
===========

``Tag.clear()`` quita los contenidos de una etiqueta::

 markup = '<a href="http://example.com/">I linked to <i>example.com</i></a>'
 soup = BeautifulSoup(markup, 'html.parser')
 tag = soup.a

 tag.clear()
 tag
 # <a href="http://example.com/"></a>

``extract()``
=============

``PageElement.extract()`` elimina una etiqueta o una cadena de caracteres
del árbol. Devuelve la etiqueta o la cadena que fue extraída::

 markup = '<a href="http://example.com/">I linked to <i>example.com</i></a>'
 soup = BeautifulSoup(markup, 'html.parser')
 a_tag = soup.a

 i_tag = soup.i.extract()

 a_tag
 # <a href="http://example.com/">I linked to</a>

 i_tag
 # <i>example.com</i>

 print(i_tag.parent)
 # None

En este punto tienes realmente dos árboles analizados: uno anclado en el
objeto :py:class:`BeautifulSoup` que usaste para analizar el documento, y
uno anclado en la etiqueta que fue extraída. Puedes llamar a ``extract``
en el hijo del elemento que extrajiste::

 my_string = i_tag.string.extract()
 my_string
 # 'example.com'

 print(my_string.parent)
 # None
 i_tag
 # <i></i>


``decompose()``
===============

``Tag.decompose()`` quita una etiqueta del árbol, y luego `lo destruye
completamente y su contenido también`::

 markup = '<a href="http://example.com/">I linked to <i>example.com</i></a>'
 soup = BeautifulSoup(markup, 'html.parser')
 a_tag = soup.a
 i_tag = soup.i

 i_tag.decompose()
 a_tag
 # <a href="http://example.com/">I linked to</a>

El comportamiento de una :py:class:`Tag` o :py:class:`NavigableString` descompuesta
no está definido y no deberías usarlo para nada. Si no estás seguro si algo
ha sido descompuesto, puedes comprobar su propiedad ``.decomposed``
`(nuevo en Beautiful Soup 4.9.0)`::

 i_tag.decomposed
 # True

 a_tag.decomposed
 # False


.. _replace_with():

``replace_with()``
==================

``PageElement.replace_with()`` elimina una etiqueta o cadena del árbol,
y lo reemplaza con una o más etiquetas de tu elección::

 markup = '<a href="http://example.com/">I linked to <i>example.com</i></a>'
 soup = BeautifulSoup(markup, 'html.parser')
 a_tag = soup.a

 new_tag = soup.new_tag("b")
 new_tag.string = "example.com"
 a_tag.i.replace_with(new_tag)

 a_tag
 # <a href="http://example.com/">I linked to <b>example.com</b></a>

 bold_tag = soup.new_tag("b")
 bold_tag.string = "example"
 i_tag = soup.new_tag("i")
 i_tag.string = "net"
 a_tag.b.replace_with(bold_tag, ".", i_tag)

 a_tag
 # <a href="http://example.com/">I linked to <b>example</b>.<i>net</i></a>


``replace_with()`` devuelve la etiqueta o cadena que se reemplazó,
así que puedes examinarla o añadirla de nuevo a otra parte del árbol.

`La capacidad de pasar múltiples argumentos a replace_with() es nueva
en Beautiful Soup 4.10.0.`


``wrap()``
==========

``PageElement.wrap()`` envuelve un elemento en la etiqueta que especificas.
Devuelve la nueva envoltura::

 soup = BeautifulSoup("<p>I wish I was bold.</p>", 'html.parser')
 soup.p.string.wrap(soup.new_tag("b"))
 # <b>I wish I was bold.</b>

 soup.p.wrap(soup.new_tag("div"))
 # <div><p><b>I wish I was bold.</b></p></div>

`Este método es nuevo en Beautiful Soup 4.0.5.`

``unwrap()``
============

``Tag.unwrap()`` es el opuesto de ``wrap()``. Reemplaza una
etiqueta con lo que haya dentro de lo que haya en esa etiqueta.
Es bueno para eliminar anotaciones::

 markup = '<a href="http://example.com/">I linked to <i>example.com</i></a>'
 soup = BeautifulSoup(markup, 'html.parser')
 a_tag = soup.a

 a_tag.i.unwrap()
 a_tag
 # <a href="http://example.com/">I linked to example.com</a>

Como ``replace_with()``, ``unwrap()`` devuelve la etiqueta que fue
reemplazada.

``smooth()``
============

Tras llamar a un puñado de métodos que modifican el árbol analizado, puedes
acabar con dos o más objetos :py:class:`NavigableString` uno al lado del otro.
Beautiful Soup no tiene ningún problema con esto, pero como no puede ocurrir
en un documento recién analizado, puedes no esperar un comportamiento como
el siguiente::

 soup = BeautifulSoup("<p>A one</p>", 'html.parser')
 soup.p.append(", a two")

 soup.p.contents
 # ['A one', ', a two']

 print(soup.p.encode())
 # b'<p>A one, a two</p>'

 print(soup.p.prettify())
 # <p>
 #  A one
 #  , a two
 # </p>

Puedes llamar a ``Tag.smooth()`` para limpiar el árbol analizado consolidando
cadenas adyacentes::

 soup.smooth()

 soup.p.contents
 # ['A one, a two']

 print(soup.p.prettify())
 # <p>
 #  A one, a two
 # </p>

`Este método es nuevo en Beautiful Soup 4.8.0.`

========
 Salida
========

.. _.prettyprinting:

*Pretty-printing*
=================

El método ``prettify()`` convertirá un árbol analizado de Beautiful Soup
en una cadena de caracteres Unicode bien formateado, con una línea
para cada etiqueta y cada cadena::

 markup = '<html><head><body><a href="http://example.com/">I linked to <i>example.com</i></a>'
 soup = BeautifulSoup(markup, 'html.parser')
 soup.prettify()
 # '<html>\n <head>\n </head>\n <body>\n  <a href="http://example.com/">\n...'

 print(soup.prettify())
 # <html>
 #  <head>
 #  </head>
 #  <body>
 #   <a href="http://example.com/">
 #    I linked to
 #    <i>
 #     example.com
 #    </i>
 #   </a>
 #  </body>
 # </html>

Puedes llamar ``prettify()`` a alto nivel sobre el objeto :py:class:`BeautifulSoup`,
o sobre cualquiera de sus objetos :py:class:`Tag`::

 print(soup.a.prettify())
 # <a href="http://example.com/">
 #  I linked to
 #  <i>
 #   example.com
 #  </i>
 # </a>

Como añade un espacio en blanco (en la forma de saltos de líneas),
``prettify()`` cambia el sentido del documento HTML y no debe ser
usado para reformatearlo. El objetivo de ``prettify()`` es ayudarte
a entender visualmente la estructura del documento en el que trabajas.
  
*Non-pretty printing*
=====================

Si tan solo quieres una cadena, sin ningún formateo adornado,
puedes llamar a ``str()`` en un objeto :py:class:`BeautifulSoup`, o
sobre una :py:class:`Tag` dentro de él::

 str(soup)
 # '<html><head></head><body><a href="http://example.com/">I linked to <i>example.com</i></a></body></html>'

 str(soup.a)
 # '<a href="http://example.com/">I linked to <i>example.com</i></a>'

La función ``str()`` devuelve una cadena codificada en UTF-8. Mira
`Codificaciones`_ para otras opciones.

Puedes también llamar a ``encode()`` para obtener un bytestring, y
``decode()`` para obtener Unicode.

.. _output_formatters:

Formatos de salida
==================

Si le das a Beautiful Soup un documento que contenga entidades HTML
como "&lquot;", serán convertidas a caracteres Unicode::

 soup = BeautifulSoup("&ldquo;Dammit!&rdquo; he said.", 'html.parser')
 str(soup)
 # '“Dammit!” he said.'

Si después conviertes el documento a bytestring, los caracteres Unicode
serán convertidos a UTF-8. No obtendrás de nuevo las entidades HTML::

 soup.encode("utf8")
 # b'\xe2\x80\x9cDammit!\xe2\x80\x9d he said.'

Por defecto, los únicos caracteres que se formatean en la salida son
ampersands y comillas anguladas simples. Estas se transforman en
"&amp;", "&lt;" y "&gt;", así Beautiful Soup no genera inadvertidamente
HTML o XML inválido::

 soup = BeautifulSoup("<p>The law firm of Dewey, Cheatem, & Howe</p>", 'html.parser')
 soup.p
 # <p>The law firm of Dewey, Cheatem, &amp; Howe</p>

 soup = BeautifulSoup('<a href="http://example.com/?foo=val1&bar=val2">A link</a>', 'html.parser')
 soup.a
 # <a href="http://example.com/?foo=val1&amp;bar=val2">A link</a>

Puedes cambiar este comportamiento dando un valor al argumento
``formatter`` de ``prettify()``, ``encode()`` o ``decode()``.
Beautiful Soup reconoce cinco posibles valores para ``formatter``.

El valor por defecto es ``formatter="minimal"``. Las cadenas solo
serán procesadas lo suficiente como para asegurar que Beautiful Soup
genera HTML/XML válido::

 french = "<p>Il a dit &lt;&lt;Sacr&eacute; bleu!&gt;&gt;</p>"
 soup = BeautifulSoup(french, 'html.parser')
 print(soup.prettify(formatter="minimal"))
 # <p>
 #  Il a dit &lt;&lt;Sacré bleu!&gt;&gt;
 # </p>

Si pasas ``formatter="html"``, Beautiful Soup convertirá caracteres
Unicode a entidades HTML cuando sea posible::

 print(soup.prettify(formatter="html"))
 # <p>
 #  Il a dit &lt;&lt;Sacr&eacute; bleu!&gt;&gt;
 # </p>

Si pasas ``formatter="html5"``, es similar a
``formatter="html"``, pero Beautiful Soup omitirá la barra de
cierre en etiquetas HTML vacías como "br"::

 br = BeautifulSoup("<br>", 'html.parser').br
 
 print(br.encode(formatter="html"))
 # b'<br/>'
 
 print(br.encode(formatter="html5"))
 # b'<br>'

Además, cualquier atributo cuyos valores son la cadena de
caracteres vacía se convertirán en atributos booleanos al
estilo HTML::

 option = BeautifulSoup('<option selected=""></option>').option
 print(option.encode(formatter="html"))
 # b'<option selected=""></option>'
 
 print(option.encode(formatter="html5"))
 # b'<option selected></option>'

*(Este comportamiento es nuevo a partir de Beautiful Soup 4.10.0.)*

Si pasas ``formatter=None``, Beautiful Soup no modificará en absoluto
las cadenas a la salida. Esta es la opción más rápida, pero puede
ocasionar que Beautiful Soup genere HTML/XML inválido, como en estos
ejemplos::

 print(soup.prettify(formatter=None))
 # <p>
 #  Il a dit <<Sacré bleu!>>
 # </p>

 link_soup = BeautifulSoup('<a href="http://example.com/?foo=val1&bar=val2">A link</a>', 'html.parser')
 print(link_soup.a.encode(formatter=None))
 # b'<a href="http://example.com/?foo=val1&bar=val2">A link</a>'

*Objetos Formatter*
-------------------

Si necesitas un control más sofisticado sobre tu salida, puedes
instanciar uno de las clases *formatters* de Beautiful Soup y pasar
dicho objeto a ``formatter``.

.. py:class:: HTMLFormatter

Usado para personalizar las reglas de formato para documentos HTML.

Aquí está el *formatter* que convierte cadenas de caracteres a mayúsculas,
como si están en un nodo de texto o en el valor de un atributo::

 from bs4.formatter import HTMLFormatter
 def uppercase(str):
     return str.upper()
 
 formatter = HTMLFormatter(uppercase)

 print(soup.prettify(formatter=formatter))
 # <p>
 #  IL A DIT <<SACRÉ BLEU!>>
 # </p>

 print(link_soup.a.prettify(formatter=formatter))
 # <a href="HTTP://EXAMPLE.COM/?FOO=VAL1&BAR=VAL2">
 #  A LINK
 # </a>

Este es el *formatter* que incrementa la sangría cuando se realiza
*pretty-printing*::

 formatter = HTMLFormatter(indent=8)
 print(link_soup.a.prettify(formatter=formatter))
 # <a href="http://example.com/?foo=val1&bar=val2">
 #         A link
 # </a>

.. py:class:: XMLFormatter

Usado para personalizar las reglas de formateo para documentos XML.

Escribir tu propio *formatter*
------------------------------

Crear una subclase a partir de :py:class:`HTMLFormatter` p :py:class:`XMLFormatter`
te dará incluso más control sobre la salida. Por ejemplo, Beautiful Soup
ordena por defecto los atributos en cada etiqueta::

 attr_soup = BeautifulSoup(b'<p z="1" m="2" a="3"></p>', 'html.parser')
 print(attr_soup.p.encode())
 # <p a="3" m="2" z="1"></p>

Para detener esto, puedes modificar en la subclase creada
el método ``Formatter.attributes()``, que controla los atributos
que se ponen en la salida y en qué orden. Esta implementación también
filtra el atributo llamado "m" cuando aparezca::

 class UnsortedAttributes(HTMLFormatter):
     def attributes(self, tag):
         for k, v in tag.attrs.items():
             if k == 'm':
                 continue
             yield k, v
 
 print(attr_soup.p.encode(formatter=UnsortedAttributes())) 
 # <p z="1" a="3"></p>

Una última advertencia: si creas un objeto :py:class:`CData`, el texto
dentro de ese objeto siempre se muestra `exactamente como aparece, sin
ningún formato`. Beautiful Soup llamará a la función de sustitución de
entidad, por si hubieses escrito una función a medida que cuenta
todas las cadenas en el documento o algo así, pero ignorará el
valor de retorno::

 from bs4.element import CData
 soup = BeautifulSoup("<a></a>", 'html.parser')
 soup.a.string = CData("one < three")
 print(soup.a.prettify(formatter="html"))
 # <a>
 #  <![CDATA[one < three]]>
 # </a>


``get_text()``
==============

Si solo necesitas el texto legible dentro de un documento o etiqueta, puedes
usar el método ``get_text()``. Devuelve todo el texto dentro del documento o
dentro de la etiqueta, como una sola cadena caracteres Unicode::

 markup = '<a href="http://example.com/">\nI linked to <i>example.com</i>\n</a>'
 soup = BeautifulSoup(markup, 'html.parser')

 soup.get_text()
 '\nI linked to example.com\n'
 soup.i.get_text()
 'example.com'

Puedes especificar una cadena que usará para unir los trozos
de texto::

 # soup.get_text("|")
 '\nI linked to |example.com|\n'

Puedes indicar a Beautiful Soup que quite los espacios en blanco del
comienzo y el final de cada trozo de texto::

 # soup.get_text("|", strip=True)
 'I linked to|example.com'

Pero en ese punto puedas querer usar mejor el generador
:ref:`.stripped_strings <string-generators>`, y procesar el texto
por tu cuenta::

 [text for text in soup.stripped_strings]
 # ['I linked to', 'example.com']

*A partir de Beautiful Soup version 4.9.0, cuando lxml o html.parser
se usan, el contenido de las etiquetas <script>, <style>, y <template>
no se consideran texto, ya que esas etiquetas no son parte de la parte
legible del contenido de la página.*

*A partir de de Beautiful Soup version 4.10.0, puedes llamar a get_text(),
.strings, o .stripped_strings en un objeto NavigableString. Devolverá
el propio objeto, o nada, así que la única razón para hacerlo es cuando
estás iterando sobre una lista mixta.*
 
==================================
 Especificar el analizador a usar
==================================

Si lo único que necesitas es analizar algún HTML, puedes ponerlo en
el constructor de :py:class:`BeautifulSoup`, y probablemente irá bien.
Beautiful Soup elegirá un analizador por ti y analizará los datos.
Pero hay algunos argumentos adicionales que puedes pasar al constructor
para cambiar el analizador que se usa.

El primer argumento del constructor de :py:class:`BeautifulSoup` es una cadena
o un gestor de archivos abierto--el marcado que quieres analizar. El segundo
argumento es `cómo` quieres que el marcado analizado.

Si no especificas nada, obtendrás el mejor analizador HTML que tengas
instalado. Beautiful Soup clasifica al analizador de lxml como el mejor,
después el de html5lib, y luego el analizador integrado en Python. Puedes
sobrescribir esto especificando uno de los siguientes:

* El tipo de marcado que quieres analizar. Actualmente se soportan
  "html", "xml", y "html5".

* El nombre de la librería del analizador que quieras usar. Actualmente se
  soportan "lxml", "html5lib", y "html.parser" (el analizador HTML integrado
  de Python).

La sección `Instalar un analizador`_ contraste los analizadores admitidos.

Si no tienes un analizador apropiado instalado, Beautiful Soup ignorará
tu petición y elegirá un analizador diferente. Ahora mismo, el único
analizador XML es lxml. Si no tienes lxml instalado, solicitar un
analizador XML no te dará uno, y pedir por "lxml" tampoco funcionará.

Diferencias entre analizadores
==============================

Beautiful Soup presenta la misma interfaz que varios analizadores,
pero cada uno es diferente. Analizadores diferentes crearán
árboles analizados diferentes a partir del mismo documento. La mayores
diferencias están entre los analizadores HTML y los XML. Este es un
documento corto, analizado como HTML usando el analizador que viene
con Python::

 BeautifulSoup("<a><b/></a>", "html.parser")
 # <a><b></b></a>

Como una sola etiqueta <b/> no es HTML válido, html.parser lo convierte a
un par <b><b/>.

Aquí está el mismo documento analizado como XML (correr esto requiere que
tengas instalado lxml). Debe notarse que la etiqueta independiente
<b/> se deja sola, y que en el documento se incluye una declaración XML
en lugar de introducirlo en una etiqueta <html>::

 print(BeautifulSoup("<a><b/></a>", "xml"))
 # <?xml version="1.0" encoding="utf-8"?>
 # <a><b/></a>

Hay también diferencias entre analizadores HTML. Si le das a Beautiful
Soup un documento HTML perfectamente formado, esas diferencias no
importan. Un analizador será más rápido que otro, pero todos te darán
una estructura de datos que será exactamente como el documento HTML
original.

Pero si el documento no está perfectamente formado, analizadores
diferentes darán diferentes resultados. A continuación se presenta
un documento corto e incorrecto analizado usando el analizador
HTML de lxml. Debe considerarse que la etiqueta <a> es envuelta
en las etiquetas <body> y <html>, y que la etiqueta colgada </p>
simplemente se ignora::

 BeautifulSoup("<a></p>", "lxml")
 # <html><body><a></a></body></html>

Este es el mismo documento analizado usando html5lib::

 BeautifulSoup("<a></p>", "html5lib")
 # <html><head></head><body><a><p></p></a></body></html>

En lugar de ignorar la etiqueta colgada </p>, html5lib la empareja
con una etiqueta inicial <p>. html5lib también añade una etiqueta <head>
vacía; lxml no se molesta.

Este es el mismo documento analizado usando el analizador HTML integrado
en Python::

 BeautifulSoup("<a></p>", "html.parser")
 # <a></a>

Como lxml, este analizador ignora la etiqueta clausura </p>.
A diferencia de html5lib o lxml, este analizador no intenta
crear un documento HTML bien formado añadiendo las etiquetas
<html> o <body>.

Como el documento "<a></p>" es inválido, ninguna de estas técnicas
es la forma 'correcta' de gestionarlo. El analizador de html5lib usa
técnicas que son parte del estándar de HTML5, así que es la que más
se puede aproximar a ser la manera correcta, pero las tres técnicas
son legítimas.

Las diferencias entre analizadores pueden afectar a tu script. Si
estás planeando en distribuir tu script con otras personas, o
ejecutarlo en varias máquinas, deberías especificar un analizador
en el constructor de :py:class:`BeautifulSoup`. Eso reducirá
las probabilidad que tus usuarios analicen un documento diferentemente
de la manera en la que tú lo analizas.

================
 Codificaciones
================

Cualquier documento HTML o XML está escrito en una codificación
específica como ASCII o UTF-8. Pero cuando cargas ese documento en
Beautiful Soup, descubrirás que se convierte en Unicode::

 markup = "<h1>Sacr\xc3\xa9 bleu!</h1>"
 soup = BeautifulSoup(markup, 'html.parser')
 soup.h1
 # <h1>Sacré bleu!</h1>
 soup.h1.string
 # 'Sacr\xe9 bleu!'

No es magia (seguro que eso sería genial). Beautiful Soup usa una
sub-librería llamada `Unicode, Dammit`_ para detectar la codificación
de un documento y convertirlo a Unicode. La codificación auto detectada
está disponible con el atributo ``.original_encoding`` del objeto
:py:class:`Beautiful Soup`::

 soup.original_encoding
 'utf-8'

Unicode, Dammit estima correctamente la mayor parte del tiempo, pero
a veces se equivoca. A veces estima correctamente, pero solo después
de una búsqueda byte a byte del documento que tarda mucho tiempo.
Si ocurre que sabes a priori la codificación del documento, puedes
evitar errores y retrasos pasándola al constructor de :py:class:`BeautifulSoup`
con ``from_encoding``.

Este es un documento escrito es ISO-8859-8. El documento es tan corto que
Unicode, Dammit no da en el clave, y lo identifica erróneamente como
ISO-8859-7::

 markup = b"<h1>\xed\xe5\xec\xf9</h1>"
 soup = BeautifulSoup(markup, 'html.parser')
 print(soup.h1)
 # <h1>νεμω</h1>
 print(soup.original_encoding)
 # iso-8859-7

Podemos arreglarlo pasándole el correcto a ``from_encoding``::

 soup = BeautifulSoup(markup, 'html.parser', from_encoding="iso-8859-8")
 print(soup.h1)
 # <h1>םולש</h1>
 print(soup.original_encoding)
 # iso8859-8

Si no sabes cuál es la codificación correcta, pero sabes que Unicode, Dammit
está suponiendo mal, puedes pasarle las opciones mal estimadas con
``exclude_encodings``::

 soup = BeautifulSoup(markup, 'html.parser', exclude_encodings=["iso-8859-7"])
 print(soup.h1)
 # <h1>םולש</h1>
 print(soup.original_encoding)
 # WINDOWS-1255

Windows-1255 no es correcto al 100%, pero esa codificación es
una superconjunto compatible con ISO-8859-8, así que se acerca
lo suficiente. (``exlcude_encodings`` es una nueva característica
en Beautiful Soup 4.4.0).

En casos raros (normalmente cuando un documento UTF-8 contiene texto
escrito en una codificación completamente diferente), la única manera
para obtener Unicode es reemplazar algunos caracteres con el carácter
Unicode especial "REPLACEMENT CHARACTER" (U+FFFD, �). Si Unicode, Dammit
necesita hacer esto, establecerá el atributo ``.contains_replacement_characters``
a ``True`` en el objeto ``UnicodeDammit`` o :py:class:`BeautifulSoup`. Esto
te permite saber si la representación Unicode no es una representación
exacta de la original--algún dato se ha perdido. Si un documento contiene �,
pero ``contains_replacement_characteres`` es ``False``, sabrás que �
estaba allí originalmente (como lo está en este párrafo) y no implica
datos perdidos.

Codificación de salida
======================

Cuando escribas completamente un documento desde Beautiful Soup,
obtienes un documento UTF-8, incluso cuando el documento no está en UTF-8
por el que empezar. Este es un documento escrito con la codificación Latin-1::

 markup = b'''
  <html>
   <head>
    <meta content="text/html; charset=ISO-Latin-1" http-equiv="Content-type" />
   </head>
   <body>
    <p>Sacr\xe9 bleu!</p>
   </body>
  </html>
 '''

 soup = BeautifulSoup(markup, 'html.parser')
 print(soup.prettify())
 # <html>
 #  <head>
 #   <meta content="text/html; charset=utf-8" http-equiv="Content-type" />
 #  </head>
 #  <body>
 #   <p>
 #    Sacré bleu!
 #   </p>
 #  </body>
 # </html>

Fíjate bien que la etiqueta <meta> ha sido reescrita para reflejar el hecho
de que el documento está ahora en UTF-8.

Si no quieres UTF-8, puedes pasar una codificación a ``prettify()``::

 print(soup.prettify("latin-1"))
 # <html>
 #  <head>
 #   <meta content="text/html; charset=latin-1" http-equiv="Content-type" />
 # ...

También puedes llamar a encode() sobre el objeto :py:class:`BeautifulSoup`, o
cualquier elemento en el objeto, como si fuese una cadena de Python::

 soup.p.encode("latin-1")
 # b'<p>Sacr\xe9 bleu!</p>'

 soup.p.encode("utf-8")
 # b'<p>Sacr\xc3\xa9 bleu!</p>'

Cualesquiera caracteres que no puedan ser representados en la codificación
que has elegido se convierten en referencias a entidades numéricas XML.
Este es un documento que incluye el carácter Unicode SNOWMAN::

 markup = u"<b>\N{SNOWMAN}</b>"
 snowman_soup = BeautifulSoup(markup, 'html.parser')
 tag = snowman_soup.b

El carácter SNOWMAN puede ser parte de un documento UTF-8 (se parece a ☃),
pero no hay representación para ese carácter en ISO-Latin-1 o ASCII,
así que se convierte en "&#9731" para esas codificaciones::

 print(tag.encode("utf-8"))
 # b'<b>\xe2\x98\x83</b>'

 print(tag.encode("latin-1"))
 # b'<b>&#9731;</b>'

 print(tag.encode("ascii"))
 # b'<b>&#9731;</b>'

Unicode, Dammit
===============

Puedes usar Unicode, Dammit sin usar Beautiful Soup. Es útil cuando
tienes datos en una codificación desconocida y solo quieres convertirlo
a Unicode::

 from bs4 import UnicodeDammit
 dammit = UnicodeDammit(b"\xc2\xabSacr\xc3\xa9 bleu!\xc2\xbb")
 print(dammit.unicode_markup)
 # «Sacré bleu!»
 dammit.original_encoding
 # 'utf-8'

Los estimaciones de Unicode, Dammit será mucho más precisas si instalas
una de estas librerías de Python: ``charset-normalizer``, ``chardet``,
o ``cchardet``. Cuanto más datos le des a Unicode, Dammit, con mayor exactitud
estimará. Si tienes alguna sospecha sobre las codificaciones que podrían ser, puedes
pasárselas en una lista::

 dammit = UnicodeDammit("Sacr\xe9 bleu!", ["latin-1", "iso-8859-1"])
 print(dammit.unicode_markup)
 # Sacré bleu!
 dammit.original_encoding
 # 'latin-1'

Unicode, Dammit tiene dos características especiales que Beautiful Soup no usa.

Comillas inteligentes
---------------------

Puedes usar Unicode, Dammit para convertir las comillas inteligentes de Microsoft
a entidades HTML o XML::

 markup = b"<p>I just \x93love\x94 Microsoft Word\x92s smart quotes</p>"

 UnicodeDammit(markup, ["windows-1252"], smart_quotes_to="html").unicode_markup
 # '<p>I just &ldquo;love&rdquo; Microsoft Word&rsquo;s smart quotes</p>'

 UnicodeDammit(markup, ["windows-1252"], smart_quotes_to="xml").unicode_markup 
 # '<p>I just &#x201C;love&#x201D; Microsoft Word&#x2019;s smart quotes</p>'

Puedes también convertir las comillas inteligentes de Microsoft a comillas ASCII::

 UnicodeDammit(markup, ["windows-1252"], smart_quotes_to="ascii").unicode_markup
 # '<p>I just "love" Microsoft Word\'s smart quotes</p>'

Con suerte encontrarás esta característica útil, pero Beautiful Soup no la usa.
Beautiful Soup prefiere el comportamiento por defecto, el cual es convertir
las comillas inteligentes de Microsoft a caracteres Unicode junto al resto
de cosas::

 UnicodeDammit(markup, ["windows-1252"]).unicode_markup
 # '<p>I just “love” Microsoft Word’s smart quotes</p>'

Codificaciones inconsistentes
-----------------------------

A veces un documento está mayoritariamente en UTF-8, pero contiene
caracteres Windows-1252 como (de nuevo) comillas inteligentes de Microsoft.
Esto puede ocurrir cuando un sitio web incluye datos de múltiples fuentes.
Puedes usar ``UnicodeDammit.detwingle()`` para convertir dicho documento en
puro UTF-8. Este un ejemplo sencillo::

 snowmen = (u"\N{SNOWMAN}" * 3)
 quote = (u"\N{LEFT DOUBLE QUOTATION MARK}I like snowmen!\N{RIGHT DOUBLE QUOTATION MARK}")
 doc = snowmen.encode("utf8") + quote.encode("windows_1252")

Este documento es un desastre. Los muñecos de nieve están en UTF-8 y las
comillas están en Windows-1252. Puedes mostrar los muñecos de nieve o
las comillas, pero no ambos::

 print(doc)
 # ☃☃☃�I like snowmen!�

 print(doc.decode("windows-1252"))
 # â˜ƒâ˜ƒâ˜ƒ“I like snowmen!”

Decodificar el documento en UTF-8 provoca un ``UnicodeDecodeError``, y
decodificarlo como Windows-1252 te da un galimatías. Afortunadamente,
``UnicodeDammit.detwingle()`` convertirá la cadena en puro UTF-8,
permitiéndote decodificarlo en Unicode y mostrar el muñeco de nieve
y marcas de comillas simultáneamente::

 new_doc = UnicodeDammit.detwingle(doc)
 print(new_doc.decode("utf8"))
 # ☃☃☃“I like snowmen!”

``UnicodeDammit.detwingle()``  solo sabe cómo gestionar Windows-1252 embebido
en UTF-8 (o viceversa, supongo), pero este es el caso más común.

Fíjate que debes saber que debes llamar a ``UnicodeDammit.detwingle()``
en tus datos antes de pasarlo a :py:class:`BeautifulSoup` o el constructor
de ``UnicodeDammit``. Beautiful Soup asume que un documento tiene una
sola codificación, la que sea. Si quieres pasar un documento que contiene
ambas UTF-8 y Windows-1252, es probable que piense que todo el documento
es Windows-1252, y el documento se parecerá a ```â˜ƒâ˜ƒâ˜ƒ“I like snowmen!”``.

``UnicodeDammit.detwingle()`` es nuevo en Beautiful Soup 4.1.0.

==================
 Números de línea
==================

Los analizadores de ``html.parser`` y ``html5lib`` pueden llevar la cuenta
de los lugares en el documento original donde se han encontrado cada etiqueta.
Puedes acceder a esta información con ``Tag.sourceline`` (número de línea) y
``Tag.sourcepos`` (posición del comienzo de una etiqueta en una línea)::

 markup = "<p\n>Paragraph 1</p>\n    <p>Paragraph 2</p>"
 soup = BeautifulSoup(markup, 'html.parser')
 for tag in soup.find_all('p'):
     print(repr((tag.sourceline, tag.sourcepos, tag.string)))
 # (1, 0, 'Paragraph 1')
 # (3, 4, 'Paragraph 2')

Debe destacarse que los dos analizadores entienden cosas ligeramente
diferentes por ``sourceline`` y ``sourcepos``. Para html.parser, estos
números representan la posición del signo "menor" inicial. Para html5lib,
estos números representan la posición del signo "mayor" final::
   
 soup = BeautifulSoup(markup, 'html5lib')
 for tag in soup.find_all('p'):
     print(repr((tag.sourceline, tag.sourcepos, tag.string)))
 # (2, 0, 'Paragraph 1')
 # (3, 6, 'Paragraph 2')

Puedes interrumpir esta característica pasado ``store_line_numbers=False``
en el constructor de :py:class:`BeautifulSoup`::

 markup = "<p\n>Paragraph 1</p>\n    <p>Paragraph 2</p>"
 soup = BeautifulSoup(markup, 'html.parser', store_line_numbers=False)
 print(soup.p.sourceline)
 # None

`Esta característica es nueva en 4.8.1, y los analizadores basados en lxml no la
soportan.`

===============================
 Comparar objetos por igualdad
===============================

Beautiful Soup indica que dos objetos :py:class:`NavigableString` o :py:class:`Tag`
son iguales cuando representan al mismo marcado HTML o XML. En este ejemplo,
las dos etiquetas <b> son tratadas como iguales, aunque están en diferentes
partes del objeto árbol, porque ambas son "<b>pizza</b>"::

 markup = "<p>I want <b>pizza</b> and more <b>pizza</b>!</p>"
 soup = BeautifulSoup(markup, 'html.parser')
 first_b, second_b = soup.find_all('b')
 print(first_b == second_b)
 # True

 print(first_b.previous_element == second_b.previous_element)
 # False

Si quieres saber si dos variables se refieren a exactamente el mismo
objeto, usa `is`::

 print(first_b is second_b)
 # False

==================================
 Copiar objetos de Beautiful Soup
==================================

Puedes usar ``copy.copy()`` para crear una copia de cualquier
:py:class:`Tag` o :py:class:`NavigableString`::

 import copy
 p_copy = copy.copy(soup.p)
 print(p_copy)
 # <p>I want <b>pizza</b> and more <b>pizza</b>!</p>

La copia se considera igual que la original, ya que representa el mismo
marcado que el original, pero no son el mismo objeto::

 print(soup.p == p_copy)
 # True

 print(soup.p is p_copy)
 # False

La única diferencia real es que la copia está completamente desconectada
del objeto árbol de Beautiful Soup, como si ``extract()`` hubiese sido
llamada sobre ella::

 print(p_copy.parent)
 # None

Esto es porque dos diferentes objetos :py:class:`Tag` no pueden ocupar
el mismo espacio al mismo tiempo.

=========================================
 Personalización avanzada del analizador
=========================================

Beautiful Soup ofrece numerosas vías para personalizar la manera en la que
el analizador trata HTML o XML entrante. Esta sección cubre las técnicas
de personalizadas usadas más comúnmente.

Analizar solo parte del documento
=================================

Digamos que quieres usar Beautiful Soup para observar las etiquetas <a> de un
documento. Es un malgasto de tiempo y memoria analizar todo el documento y
después recorrerlo una y otra vez buscando etiquetas <a>. Sería mucho más
rápido ignorar todo lo que no sea una etiqueta <a> desde el principio.
La clase :py:class:`SoupStrainer` te permite elegir qué partes de un
documento entrante se analizan. Tan solo crea un :py:class:`SoupStrainer` y
pásalo al constructor de :py:class:`BeautifulSoup` en el argumento ``parse_only``.

(Debe notarse que *esta característica no funcionará si estás usando el
analizador de html5lib*. Si usas html5lib, todo el documento será analizado,
no importa el resto. Esto es porque html5lib constantemente reorganiza el
árbol analizado conforme trabaja, y si alguna parte del documento no
consigue introducirse en el árbol analizado, se quedará colgado. Para evitar
confusión en los ejemplos más abajo forzaré a Beautiful Soup a que use
el analizador integrado de Python).

.. py:class:: SoupStrainer

La clase :py:class:`SoupStrainer` toma los mismos argumentos que un típico
método de `Buscar en el árbol`_: :ref:`name <name>`, :ref:`attrs <attrs>`,
:ref:`string <string>`, y :ref:`**kwargs <kwargs>`. Estos son tres objetos
:py:class:`SoupStrainer`::

 from bs4 import SoupStrainer

 only_a_tags = SoupStrainer("a")

 only_tags_with_id_link2 = SoupStrainer(id="link2")

 def is_short_string(string):
     return string is not None and len(string) < 10

 only_short_strings = SoupStrainer(string=is_short_string)

Voy a traer de nuevo el documento de "Las tres hermanas" una vez más,
y veremos cómo parece el documento cuando es analizado con estos
tres objetos :py:class:`SoupStrainer`::

 html_doc = """<html><head><title>The Dormouse's story</title></head>
 <body>
 <p class="title"><b>The Dormouse's story</b></p>

 <p class="story">Once upon a time there were three little sisters; and their names were
 <a href="http://example.com/elsie" class="sister" id="link1">Elsie</a>,
 <a href="http://example.com/lacie" class="sister" id="link2">Lacie</a> and
 <a href="http://example.com/tillie" class="sister" id="link3">Tillie</a>;
 and they lived at the bottom of a well.</p>

 <p class="story">...</p>
 """

 print(BeautifulSoup(html_doc, "html.parser", parse_only=only_a_tags).prettify())
 # <a class="sister" href="http://example.com/elsie" id="link1">
 #  Elsie
 # </a>
 # <a class="sister" href="http://example.com/lacie" id="link2">
 #  Lacie
 # </a>
 # <a class="sister" href="http://example.com/tillie" id="link3">
 #  Tillie
 # </a>

 print(BeautifulSoup(html_doc, "html.parser", parse_only=only_tags_with_id_link2).prettify())
 # <a class="sister" href="http://example.com/lacie" id="link2">
 #  Lacie
 # </a>

 print(BeautifulSoup(html_doc, "html.parser", parse_only=only_short_strings).prettify())
 # Elsie
 # ,
 # Lacie
 # and
 # Tillie
 # ...
 #

Puedes también pasar un :py:class:`SoupStrainer` en cualquiera de los métodos
cubiertos en `Buscar en el árbol`_. Esto probablemente no sea terriblemente útil,
pero pensé en mencionarlo::

 soup = BeautifulSoup(html_doc, 'html.parser')
 soup.find_all(only_short_strings)
 # ['\n\n', '\n\n', 'Elsie', ',\n', 'Lacie', ' and\n', 'Tillie',
 #  '\n\n', '...', '\n']

Personalizar atributos multivaluados
====================================

En un documento HTML, a un atributo como ``class`` se le da una lista
de valores, y a un atributo como ``id`` se le da un solo valor, porque
la especificación de HTML trata a esos atributos de manera diferente::

 markup = '<a class="cls1 cls2" id="id1 id2">'
 soup = BeautifulSoup(markup, 'html.parser')
 soup.a['class']
 # ['cls1', 'cls2']
 soup.a['id']
 # 'id1 id2'

Puedes interrumpir esto pasando ``multi_values_attributes=None``. Entonces
a todos los atributos se les dará un solo valor::

 soup = BeautifulSoup(markup, 'html.parser', multi_valued_attributes=None)
 soup.a['class']
 # 'cls1 cls2'
 soup.a['id']
 # 'id1 id2'

Puedes personalizar este comportamiento un poco pasando un diccionario
a ``multi_values_attributes``. Si lo necesitas, échale un vistazo a
``HTMLTreeBuilder.DEFAULT_CDATA_LIST_ATTRIBUTES`` para ver la configuración
que Beautiful Soup usa por defecto, que está basada en la especificación
HTML.

`(Esto es una nueva característica en Beautiful Soup 4.8.0).`

Gestionar atributos duplicados
==============================

Cuando se use el analizador de ``html.parser``, puedes usar
el argumento del constructor ``on_duplicate_attribute`` para personalizar
qué hace Beautiful Soup cuando encuentra una etiqueta que define el mismo
atributo más de una vez::

 markup = '<a href="http://url1/" href="http://url2/">'

El comportamiento por defecto es usar el último valor encontrado en la
etiqueta::

 soup = BeautifulSoup(markup, 'html.parser')
 soup.a['href']
 # http://url2/

 soup = BeautifulSoup(markup, 'html.parser', on_duplicate_attribute='replace')
 soup.a['href']
 # http://url2/

Con ``on_duplicate_attribute='ignore'`` puedes indicar a Beautiful Soup que
use el `primer` valor encontrado e ignorar el resto::

 soup = BeautifulSoup(markup, 'html.parser', on_duplicate_attribute='ignore')
 soup.a['href']
 # http://url1/

(lxml y html5lib siempre lo hacen así; su comportamiento no puede ser
configurado desde Beautiful Soup.)

Si necesitas más, puedes pasar una función que sea llamada en cada valor duplicado::

 def accumulate(attributes_so_far, key, value):
     if not isinstance(attributes_so_far[key], list):
         attributes_so_far[key] = [attributes_so_far[key]]
     attributes_so_far[key].append(value)

 soup = BeautifulSoup(markup, 'html.parser', on_duplicate_attribute=accumulate)
 soup.a['href']
 # ["http://url1/", "http://url2/"]


`(Esto es una nueva característica en Beautiful Soup 4.9.1.)`

Instanciar subclases personalizadas
===================================

Cuando un analizador indica a Beautiful Soup sobre una etiqueta o una cadena,
Beautiful Soup instanciará un objeto :py:class:`Tag` o :py:class:`NavigableString`
para contener esa información. En lugar de ese comportamiento por defecto,
puedes indicar a Beautiful Soup que instancia `subclases` de :py:class:`Tag` o
:py:class:`NavigableString`, subclases que defines con comportamiento
personalizado::

 from bs4 import Tag, NavigableString
 class MyTag(Tag):
     pass


 class MyString(NavigableString):
     pass


 markup = "<div>some text</div>"
 soup = BeautifulSoup(markup, 'html.parser')
 isinstance(soup.div, MyTag)
 # False
 isinstance(soup.div.string, MyString)
 # False 

 my_classes = { Tag: MyTag, NavigableString: MyString }
 soup = BeautifulSoup(markup, 'html.parser', element_classes=my_classes)
 isinstance(soup.div, MyTag)
 # True
 isinstance(soup.div.string, MyString)
 # True  


Esto puede ser útil cuando se incorpore Beautiful Soup en un *framework*
de pruebas.

`(Esto es una nueva característica de Beautiful Soup 4.8.1.)`

=========================
 Resolución de problemas
=========================

.. _diagnose:

``diagnose()``
==============

Si estás teniendo problemas para entender qué hace Beautiful Soup a un
documento, pasa el documento a la función ``diagnose()``. (Nuevo en
Beautiful Soup 4.2.0) Beautiful Soup imprimirá un informe mostrándote
cómo manejan el documento diferentes analizadores, y te dirán si
te falta un analizador que Beautiful Soup podría estar usando::

 from bs4.diagnose import diagnose
 with open("bad.html") as fp:
     data = fp.read()

 diagnose(data)

 # Diagnostic running on Beautiful Soup 4.2.0
 # Python version 2.7.3 (default, Aug  1 2012, 05:16:07)
 # I noticed that html5lib is not installed. Installing it may help.
 # Found lxml version 2.3.2.0
 #
 # Trying to parse your data with html.parser
 # Here's what html.parser did with the document:
 # ...

Tan solo mirando a la salida de diagnose() puede mostrate cómo resolver
el problema. Incluso si no, puedes pegar la salida de ``diagnose()``
cuando pidas ayuda.

Errores analizando un documento
===============================

Hay dos tipos diferentes de errores de análisis. Hay veces en que
se queda colgado, donde le das a Beautiful Soup un documento y
lanza una excepción, normalmente un ``HTMLParser.HTMLParseError``. Y hay
comportamientos inesperados, donde un árbol analizado de Beautiful Soup
parece muy diferente al documento usado para crearlo.

Casi ninguno de estos problemas resultan ser problemas con Beautiful Soup.
Esto no es porque Beautiful Soup sea una increíble y bien escrita pieza
de software. Es porque Beautiful Soup no incluye ningún código de
análisis. En lugar de eso, depende de análisis externos. Si un analizador
no está funcionando en un documento concreto, la mejor solución es probar
con otro analizador. Échale un vistazo a `Instalar un analizador`_ para
detalles y una comparativa de analizadores.

Los errores de análisis más comunes son ``HTMLParser.HTMLParseError:
malformed start tag`` y ``HTMLParser.HTMLParseError: bad end
tag``. Ambos son generados por la librería del analizador HTML
incluido en Python, y la solución es :ref:`instalar lxml o html5lib.
<parser-installation>`

El comportamiento inesperado más común es que no puedas encontrar
una etiqueta que sabes que está en el documento. La viste llegar, pero
``find_all()`` devuelve ``[]`` o ``find()`` devuelve ``None``. Esto
es otro problema común con el analizador HTML integrado en Python, el cual
a veces omite etiquetas que no entiende. De nuevo, la mejor solución es
:ref:`instalar lxml o html5lib. <parser-installation>`.

Problemas de incompatibilidad de versiones
==========================================

* ``SyntaxError: Invalid syntax`` (on the line ``ROOT_TAG_NAME =
  '[document]'``): Causado por ejecutar una version antigua de Beautiful
  Soup de Python 2 bajo Python 3, sin convertir el código.

* ``ImportError: No module named HTMLParser`` - Causado por ejecutar
  una version antigua de Beautiful Soup de Python 2 bajo Python 3.

* ``ImportError: No module named html.parser`` - Causado por ejecutar
  una version de Beautiful Soup de Python 3 bajo Python 2.

* ``ImportError: No module named BeautifulSoup`` - Causado por ejecutar
  código de Beautiful Soup 3 en un sistema que no tiene BS3 instalado. O
  al escribir código de Beautiful Soup 4 sin saber que el nombre del paquete
  se cambió a ``bs4``.

* ``ImportError: No module named bs4`` - Causado por ejecutar código de
  Beautiful Soup 4 en un sistema que no tiene BS4 instalado.

.. _parsing-xml:

Analizar XML
============

Por defecto, Beautiful Soup analiza documentos HTML. Para analizar
un documento como XML, pasa "xml" como el segundo argumento al
constructor :py:class:`BeautifulSoup`::

 soup = BeautifulSoup(markup, "xml")

Necesitarás :ref:`tener lxml instalado <parser-installation>`.

Otros problemas de análisis
===========================

* Si tu script funciona en un ordenador pero no en otro, o en un
  entorno virtual pero no en otro, o fuera del entorno virtual
  pero no dentro, es probable porque los dos entornos tienen
  diferentes librerías de analizadores disponibles. Por ejemplo,
  puedes haber desarrollado el script en un ordenador que solo
  tenga html5lib instalado. Mira `Diferencias entre analizadores`_
  por qué esto importa, y solucionar el problema especificando una
  librería de análisis en el constructor de :py:class:`Beautiful Soup`.

* Porque `las etiquetas y atributos de HTML son sensibles a mayúsculas
  y minúsculas <http://www.w3.org/TR/html5/syntax.html#syntax>`_,
  los tres analizadores HTML convierten los nombres de las etiquetas y
  atributos a minúscula. Esto es, el marcado <TAG></TAG> se convierte
  a <tag></tag>. Si quieres preservar la mezcla entre minúscula y
  mayúscula o mantener las mayúsculas en etiquetas y atributos,
  necesitarás :ref:`analizar el documento como XML. <parsing-xml>`

.. _misc:

Diversos
========

* ``UnicodeEncodeError: 'charmap' codec can't encode character
  '\xfoo' in position bar`` (o cualquier otro 
  ``UnicodeEncodeError``) - Este problema aparece principalmente
  en dos situaciones. Primero, cuando intentas mostrar un carácter
  Unicode que tu consola no sabe cómo mostrar (mira `esta página en la
  wiki de Python <http://wiki.python.org/moin/PrintFails>`_). Segundo,
  cuando estás escribiendo en un archivo y pasas un carácter Unicode
  que no se soporta en tu codificación por defecto. En este caso,
  la solución más simple es codificar explícitamente la cadena Unicode
  en UTF-8 con ``u.encode("utf8")``.

* ``KeyError: [attr]`` - Causado por acceder a ``tag['attr']`` cuando
  la etiqueta en cuestión no define el atributo ``'attr'``. Los
  errores más comunes son ``KeyError: 'href'`` y ``KeyError: 'class``.
  Usa ``tag.get('attr')`` si no estás seguro si ``attr`` está definido,
  tal y como harías con un diccionario de Python.

* ``AttributeError: 'ResultSet' object has no attribute 'foo'`` - Esto
  normalmente ocurre cuando esperas que ``find_all()`` devuelva
  una sola etiqueta o cadena. Pero ``find_all()`` devuelve una
  `lista` de etiquetas y cadenas--un objeto ``ResultSet``. Tienes que
  iterar sobre la lista y comprobar el ``.foo`` de cada uno, O, si solo
  quieres un resultado, tienes que usar ``find()`` en lugar de
  ``find_all()``. 

* ``AttributeError: 'NoneType' object has no attribute 'foo'`` - Esto
  normalmente ocurre porque llamaste a ``find()`` y después intentaste
  acceder al atributo ``.foo`` del resultado. Pero en tu caso, ``find()``
  no encontró nada, así que devolvió ``None``, en lugar de devolver
  una etiqueta o una cadena de caracteres. Necesitas averiguar por qué
  ``find()`` no está devolviendo nada.

* ``AttributeError: 'NavigableString' object has no attribute
  'foo'`` - Esto ocurre normalmente porque estás tratando una
  cadena de caracteres como si fuese una etiqueta. Puedes estar iterando
  sobre una lista, esperando que tan solo contenga etiquetas, pero en
  realidad contiene tanto etiquetas como cadenas.


Mejorar el rendimiento
======================

Beautiful Soup nunca será tan rápido como los analizadores en los que
se basa. Si el tiempo de respuesta es crítico, si estás pagando por
tiempo de uso por hora, o si hay alguna otra razón por la que el tiempo
de computación es más valioso que el tiempo del programador, deberías
olvidarte de Beautiful Soup y trabajar directamente sobre
`lxml <http://lxml.de/>`_.

Dicho esto, hay cosas que puedes hacer para aumentar la velocidad de
Beautiful Soup. Si no estás usando lxml como el analizador que hay
por debajo, mi consejo es que :ref:`empieces a usarlo <parser-installation>`.
Beautiful Soup analiza documentos significativamente más rápido usando
lxml que usando html.parser o html5lib.

Puedes aumentar la velocidad de detección de codificación significativamente
instalando la librería `cchardet <http://pypi.python.org/pypi/cchardet/>`_.

`Analizar solo parte del documento`_ no te ahorrará mucho tiempo de análisis, pero puede
ahorrar mucha memoria, y hará que `buscar` en el documento sea mucho más rápido.

==============================
 Traducir esta documentación
==============================

Nuevas traducciones de la documentación de Beautiful Soup se agradecen
enormemente. Las traducciones deberían estar bajo la licencia del MIT, tal
y como están Beautiful Soup y su documentación en inglés.

Hay dos maneras para que tu traducción se incorpore a la base de código
principal y al sitio de Beautiful Soup:

1. Crear una rama del repositorio de Beautiful Soup, añadir tus
   traducciones, y proponer una fusión (*merge*) con la rama principal, lo
   mismo que se haría con una propuesta de código del código fuente.

2. Enviar un mensaje al grupo de discusión de Beautiful Soup con un
   enlace a tu traducción, o adjuntar tu traducción al mensaje.

Utiliza la traducción china o portugués-brasileño como tu modelo. En
particular, por favor, traduce el archivo fuente ``doc/source/index.rst``,
en vez de la versión HTML de la documentación. Esto hace posible que la
documentación se pueda publicar en una variedad de formatos, no solo HTML.

==================
 Beautiful Soup 3
==================

Beautiful Soup 3 es la serie de lanzamientos anterior, y no está siendo
activamente desarrollada. Actualmente está empaquetada con las
distribuciones de Linux más grandes:

:kbd:`$ apt-get install python-beautifulsoup`

También está publicada a través de PyPI como :py:class:`BeautifulSoup`.:

:kbd:`$ easy_install BeautifulSoup`

:kbd:`$ pip install BeautifulSoup`

También puedes `descargar un tarball de Beautiful Soup 3.2.0
<http://www.crummy.com/software/BeautifulSoup/bs3/download/3.x/BeautifulSoup-3.2.0.tar.gz>`_.

Si ejecutaste ``easy_install beautifulsoup`` o ``easy_install BeautifulSoup``,
pero tu código no funciona, instalaste por error Beautiful Soup 3. Necesitas
ejecutar ``easy_install beautifulsoup4``.

`La documentación de Beautiful Soup 3 está archivada online
<http://www.crummy.com/software/BeautifulSoup/bs3/documentation.html>`_.

Actualizar el código a BS4
==========================

La mayoría del código escrito con Beautiful Soup 3 funcionará
con Beautiful Soup 4 con un cambio simple. Todo lo que debes hacer
es cambiar el nombre del paquete de :py:class:`BeautifulSoup` a
``bs4``. Así que esto::

 from BeautifulSoup import BeautifulSoup

se convierte en esto::

 from bs4 import BeautifulSoup

* Si obtienes el ``ImportError`` "No module named BeautifulSoup`", tu
  problema es que estás intentando ejecutar código de Beautiful Soup 3,
  pero solo tienes instalado Beautiful Soup 4.

* Si obtienes el ``ImportError`` "No module named bs4", tu problema
  es que estás intentando ejecutar código Beautiful Soup 4, pero solo
  tienes Beautiful Soup 3 instalado.

Aunque BS4 es mayormente compatible con la versión anterior BS3, la
mayoría de sus métodos han quedado obsoletos y dados nuevos nombres
para que `cumplan con PEP 8 <http://www.python.org/dev/peps/pep-0008/>`_.
Hay muchos otros renombres y cambios, y algunos de ellos rompen
con la compatibilidad hacia atrás.

Esto es todo lo que necesitarás saber para convertir tu código y hábitos BS3 a
BS4:

Necesitas un analizador
-----------------------

Beautiful Soup 3 usaba el ``SGMLParser`` de Python, un módulo que
fue obsoleto y quitado en Python 3.0. Beautiful Soup 4 usa
``html.parser`` por defecto, pero puedes conectar lxml o html5lib
y usar esos. Mira `Instalar un analizador`_ para una comparación.

Como ``html.parser`` no es el mismo analizador que ``SGMLParser``,
podrías encontrarte que Beautiful Soup 4 te de un árbol analizado
diferente al que te da Beautiful Soup 3 para el mismo marcado. Si
cambias ``html.parser`` por lxml o html5lib, puedes encontrarte
que el árbol analizado también cambia. Si esto ocurre, necesitarás
actualizar tu código de *scraping* para gestionar el nuevo árbol.

Nombre de los métodos
---------------------

* ``renderContents`` -> ``encode_contents``
* ``replaceWith`` -> ``replace_with``
* ``replaceWithChildren`` -> ``unwrap``
* ``findAll`` -> ``find_all``
* ``findAllNext`` -> ``find_all_next``
* ``findAllPrevious`` -> ``find_all_previous``
* ``findNext`` -> ``find_next``
* ``findNextSibling`` -> ``find_next_sibling``
* ``findNextSiblings`` -> ``find_next_siblings``
* ``findParent`` -> ``find_parent``
* ``findParents`` -> ``find_parents``
* ``findPrevious`` -> ``find_previous``
* ``findPreviousSibling`` -> ``find_previous_sibling``
* ``findPreviousSiblings`` -> ``find_previous_siblings``
* ``getText`` -> ``get_text``
* ``nextSibling`` -> ``next_sibling``
* ``previousSibling`` -> ``previous_sibling``

Algunos argumentos del constructor de Beautiful Soup fueron renombrados
por la misma razón:

* ``BeautifulSoup(parseOnlyThese=...)`` -> ``BeautifulSoup(parse_only=...)``
* ``BeautifulSoup(fromEncoding=...)`` -> ``BeautifulSoup(from_encoding=...)``

Renombré un método para compatibilidad con Python 3:

* ``Tag.has_key()`` -> ``Tag.has_attr()``

Renombré un atributo para usar terminología más precisa:

* ``Tag.isSelfClosing`` -> ``Tag.is_empty_element``

Renombré tres atributos para evitar usar palabras que tienen un significado
especial en Python. A diferencia de otros, estos cambios no soportan
*compatibilidad hacia atrás*. Si usaste estos atributos en BS3, tu código
se romperá en BS4 hasta que lo cambies.

* ``UnicodeDammit.unicode`` -> ``UnicodeDammit.unicode_markup``
* ``Tag.next`` -> ``Tag.next_element``
* ``Tag.previous`` -> ``Tag.previous_element``

Estos métodos sobras desde la API de Beautiful Soup 2. Han quedado
obsoletos desde 2006, y no deberían usarse en absoluto:

* ``Tag.fetchNextSiblings``
* ``Tag.fetchPreviousSiblings``
* ``Tag.fetchPrevious``
* ``Tag.fetchPreviousSiblings``
* ``Tag.fetchParents``
* ``Tag.findChild``
* ``Tag.findChildren``


Generadores
-----------

Le di a los generadores nombres que cumplan con PEP 8, y se transformaron
en propiedades:

* ``childGenerator()`` -> ``children``
* ``nextGenerator()`` -> ``next_elements``
* ``nextSiblingGenerator()`` -> ``next_siblings``
* ``previousGenerator()`` -> ``previous_elements``
* ``previousSiblingGenerator()`` -> ``previous_siblings``
* ``recursiveChildGenerator()`` -> ``descendants``
* ``parentGenerator()`` -> ``parents``

Así que en lugar de esto::

 for parent in tag.parentGenerator():
     ...

Puedes escribir esto::

 for parent in tag.parents:
     ...

(Pero el código antiguo seguirá funcionando).

Algunos de los generadores solían devolver ``None`` después de que hayan
terminado, y después paran. Eso era un error. Ahora el generador tan solo
para.

Hay dos nuevos generadores, :ref:`.strings y .stripped_strings
<string-generators>`. ``.strings`` devuelve objetos NavigableString,
y ``.stripped_strings`` devuelve cadenas de Python cuyos espacios
en blanco al comienzo y al final han sido quitados.

XML
---

Ya no hay una clase ``BeautifulStoneSoup`` para analizar XML. Para
analizar XML pasas "xml" como el segundo argumento del constructor
de :py:class:`BeautifulSoup`. Por la misma razón, el constructor
de :py:class:`BeautifulSoup` ya no reconoce el argumento ``isHTML``.

La gestión de Beautiful Soup sobre las etiquetas XML sin elementos ha sido
mejorada. Previamente cuando analizabas XML tenías que indicar
explícitamente qué etiquetas eran consideradas etiquetas sin elementos.
El argumento ``selfClosingTags`` al constructor ya no se reconoce.
En lugar de ello, Beautiful Soup considera cualquier etiqueta vacía como
una etiqueta sin elementos. Si añades un hijo a una etiqueta sin elementos,
deja de ser una etiqueta sin elementos.

Entidades
---------

Una entidad HTML o XML entrante siempre se convierte al correspondiente
carácter Unicode. Beautiful Soup 3 tenía varias formas solapadas para
gestionar entidades, las cuales se han eliminado. El constructor de
:py:class:`BeautifulSoup` ya no reconoce los argumentos ``smartQuotesTo``
o ``convertEntities`` (`Unicode, Dammit`_ aún tiene ``smart_quotes_to``,
pero por defecto ahora transforma las comillas inteligentes a Unicode).
Las constantes ``HTML_ENTITIES``, ``XML_ENTITIES``, y ``XHTML_ENTITIES``
han sido eliminadas, ya que configuran una característica (transformando
algunas pero no todas las entidades en caracteres Unicode) que ya no
existe.

Si quieres volver a convertir caracteres Unicode en entidades HTML
a la salida, en lugar de transformarlos a caracteres UTF-8, necesitas
usar un :ref:`*formatter* de salida <output_formatters>`.

Otro
----

:ref:`Tag.string <.string>` ahora funciona recursivamente. Si una
etiqueta A contiene una sola etiqueta B y nada más, entonces
A.string es el mismo que B.string (Antes, era ``None``).

Los `atributos multivaluados`_ como ``class`` tienen listas de cadenas
de caracteres como valores, no cadenas. Esto podría afectar la manera
en la que buscas por clases CSS.

Objetos :py:class:`Tag` ahora implementan el método ``__hash__``, de tal
manera que dos objetos :py:class:`Tag` se consideran iguales si generan
el mismo marcado. Esto puede cambiar el comportamiento de tus scripts
si insertas los objetos :py:class:`Tag` en un diccionario o conjunto.

Si pasas a unos de los métodos ``find*`` una :ref:`cadena <string>` y
un argumento específico de una etiqueta como :ref:`name <name>`, Beautiful
Soup buscará etiquetas que casen con tu criterio específico de la etiqueta
y cuyo :ref:`Tag.string <.string>` case con tu valor para la :ref:`cadena <string>`.
`No` encontrará las cadenas mismas. Anteriormente, Beautiful Soup ignoraba el
argumento específico de la etiqueta y buscaba por cadenas de caracteres.

El constructor de :py:class:`Beautiful Soup` ya no reconoce el argumento
`markupMassage`. Es ahora responsabilidad del analizador gestionar el marcado
correctamente.

Los analizadores alternativos, que rara vez se utilizaban, como
``ICantBelieveItsBeautifulSoup`` y ``BeautifulSOAP`` se han eliminado.
Ahora es decisión del analizador saber cómo gestionar marcado ambiguo.

El método ``prettify()`` ahora devuelve una cadena Unicode, no un bytestring.
