"""Demonstrate how different parsers parse the same markup.

Beautiful Soup can use any of a number of different parsers. Every
parser should behave more or less the same on valid markup, and
Beautiful Soup's unit tests make sure this is the case. But every
parser handles invalid markup differently. Even different versions of
the same parser handle invalid markup differently. So instead of unit
tests I've created this educational demonstration script.

The DEMO_MARKUP variable below contains many lines of HTML. This
script tests each line of markup against every parser you have
installed, and prints out how each parser sees that markup. This may
help you choose a parser, or understand why Beautiful Soup presents
your document the way it does.
"""

DEMO_MARKUP = """A bare string
<!DOCTYPE xsl:stylesheet SYSTEM "htmlent.dtd">
<!DOCTYPE xsl:stylesheet PUBLIC "htmlent.dtd">
<div><![CDATA[A CDATA section where it doesn't belong]]></div>
<div><svg><![CDATA[HTML5 does allow CDATA sections in SVG]]></svg></div>
<div>A <meta> tag</div>
<div>A <br> tag that supposedly has contents.</br></div>
<div>AT&T</div>
<div><textarea>Within a textarea, markup like <b> tags and <&<&amp; should be treated as literal</textarea></div>
<div><script>if (i < 2) { alert("<b>Markup within script tags should be treated as literal.</b>"); }</script></div>
<div>This numeric entity is missing the final semicolon: <x t="pi&#241ata"></div>
<div><a href="http://example.com/</a> that attribute value never got closed</div>
<div><a href="foo</a>, </a><a href="bar">that attribute value was closed by the subsequent tag</a></div>
<! This document starts with a bogus declaration ><div>a</div>
<div>This document contains <!an incomplete declaration <div>(do you see it?)</div>
<div>This document ends with <!an incomplete declaration
<div><a style={height:21px;}>That attribute value was bogus</a></div>
<! DOCTYPE html PUBLIC "-//W3C//DTD XHTML 1.0 Transitional//EN">The doctype is invalid because it contains extra whitespace
<div><table><td nowrap>That boolean attribute had no value</td></table></div>
<div>Here's a nonexistent entity: &#foo; (do you see it?)</div>
<div>This document ends before the entity finishes: &gt
<div><p>Paragraphs shouldn't contain block display elements, but this one does: <dl><dt>you see?</dt></p>
<b b="20" a="1" b="10" a="2" a="3" a="4">Multiple values for the same attribute.</b>
<div><table><tr><td>Here's a table</td></tr></table></div>
<div><table id="1"><tr><td>Here's a nested table:<table id="2"><tr><td>foo</td></tr></table></td></div>
<div>This tag contains nothing but whitespace: <b>    </b></div>
<div><blockquote><p><b>This p tag is cut off by</blockquote></p>the end of the blockquote tag</div>
<div><table><div>This table contains bare markup</div></table></div>
<div><div id="1">\\n <a href="link1">This link is never closed.\\n</div>\\n<div id="2">\\n <div id="3">\\n   <a href="link2">This link is closed.</a>\\n  </div>\\n</div></div>
<div>This document contains a <!DOCTYPE surprise>surprise doctype</div>
<div><a><B><Cd><EFG>Mixed case tags are folded to lowercase</efg></CD></b></A></div>
<div><our☃>Tag name contains Unicode characters</our☃></div>
<div><a ☃="snowman">Attribute name contains Unicode characters</a></div>
<!DOCTYPE html PUBLIC "-//W3C//DTD XHTML 1.0 Transitional//EN" "http://www.w3.org/TR/xhtml1/DTD/xhtml1-transitional.dtd">"""

from io import StringIO
import sys
from bs4 import BeautifulSoup
parsers = ['html.parser']

try:
    from bs4.builder import _lxml
    parsers.append('lxml')
except ImportError:
    pass

try:
    from bs4.builder import _html5lib
    parsers.append('html5lib')
except ImportError:
    pass

class Demonstration(object):
    def __init__(self, markup):
        self.results = {}
        self.markup = markup

    def run_against(self, *parser_names):
        uniform_results = True
        previous_output = None
        for parser in parser_names:
            try:
                soup = BeautifulSoup(self.markup, parser)
                if markup.startswith("<div>"):
                    # Extract the interesting part
                    output = soup.div
                else:
                    output = soup
            except Exception as e:
                output = "[EXCEPTION] %s" % str(e)
            self.results[parser] = output
            if previous_output is None:
                previous_output = output
            elif previous_output != output:
                uniform_results = False
        return uniform_results

    def dump(self):
        print("%s: %s" % ("Markup".rjust(13), self.markup))
        for parser, output in self.results.items():
            print("%s: %s" % (parser.rjust(13), output))

different_results = []
uniform_results = []

print("= Testing the following parsers: %s =" % ", ".join(parsers))
print()

input_file = sys.stdin
if sys.stdin.isatty():
    input_file = StringIO(DEMO_MARKUP)

for markup_line in input_file.readlines():
    markup = markup_line.strip().replace("\\n", "\n")
    demo = Demonstration(markup)
    is_uniform = demo.run_against(*parsers)
    if is_uniform:
        uniform_results.append(demo)
    else:
        different_results.append(demo)

print("== Markup that's handled the same in every parser ==")
print()
for demo in uniform_results:
    demo.dump()
    print()
print("== Markup that's not handled the same in every parser ==")
print()
for demo in different_results:
    demo.dump()
    print()
