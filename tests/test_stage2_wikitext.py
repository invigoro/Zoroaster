import unittest

from src.stage2.wikitext import plain_text, prose


class PlainTextTest(unittest.TestCase):
    def test_prose_keeps_link_labels_and_drops_references(self):
        self.assertEqual(plain_text(
            "After the signing of the Phase One agreement, the [[Global Times]] published a series of articles "
            "on the [[Trade war|trade war]].<ref name=\":Mao\">{{cite news |title=X |url=http://x.cn}}</ref> "
            "It ended.<ref name=\"a\" />"
        ), "After the signing of the Phase One agreement, the Global Times published a series of articles "
           "on the trade war. It ended.")

    def test_span_cut_inside_markup(self):
        # ends inside a reference
        self.assertEqual(plain_text('Crooks was a registered Republican;<ref name="J" /><ref>{{Cite web | first1=Ryan'),
                         "Crooks was a registered Republican;")
        # starts inside a citation, then only markup
        self.assertEqual(plain_text("|pages=146-164}}</ref>{{rp|148, 151}}&nbsp;<ref>Motherhood"), "")
        # starts inside a link, ends inside another
        self.assertEqual(plain_text("offense|Not Good]] and later moved to [[New York City|New York"),
                         "Not Good and later moved to New York")
        self.assertEqual(plain_text("It was sold. [[Category:Living peop"), "It was sold.")
        # starts inside an infobox or table
        self.assertEqual(plain_text("| population = 12,345\n| area = 5\n}}\nThe town grew."), "The town grew.")
        self.assertEqual(plain_text("|name=[[Joshwa Campbell]]|age={{bda|2006|2|15}}|caps=3"), "")

    def test_templates_tables_and_files_go(self):
        self.assertEqual(plain_text("{{short description|Type of US airline (1938-1978)}}"), "")
        self.assertEqual(plain_text(
            "Intro. [[File:Brody Lamb.jpg|thumb|upright|[[Brody Lamb]] was selected 104th.]] "
            "Lamb was drafted by the [[New York Rangers|Rangers]].{{citation needed|date={{CURRENTYEAR}}}}"
        ), "Intro. Lamb was drafted by the Rangers.")
        self.assertEqual(plain_text(
            '{| class="wikitable"\n|-\n! Name !! Class\n|-\n| [[Foo]] || Bar\n|}\nThe fleet grew in 2024.'
        ), "The fleet grew in 2024.")

    def test_headings_lists_formatting_and_html(self):
        self.assertEqual(plain_text(
            "==Illumination==\n* '''Bold''' item with [http://example.com a link] and http://bare.example.org\n"
            "<!-- hidden --># ''Second'' item&nbsp;&ndash; done<br />"
        ), "Illumination Bold item with a link and Second item – done")
        self.assertEqual(plain_text('On Earth<span class="anchor" id="Terrestrial"></span>'), "On Earth")
        self.assertEqual(plain_text("===DNEG Animation===\n===== 2020s =====\nText"), "DNEG Animation 2020s Text")


class ProseTest(unittest.TestCase):
    def test_leftover_parameters_and_table_cells_are_not_prose(self):
        self.assertEqual(prose("Monde |archive-date=29 August 2024"), ["Monde"])  # inside a citation
        self.assertEqual(prose('sticky-header" |+Parliamentary constituencies in the East Midlands, 2024'),
                         ['sticky-header"', "Parliamentary constituencies in the East Midlands, 2024"])
        self.assertEqual(prose("2019|| |- |||1,234 5,678"), [])
        self.assertEqual(prose("He scored on May 12.<ref>{{cite web|title=Hu}}</ref>"), ["He scored on May 12."])


if __name__ == "__main__":
    unittest.main()
