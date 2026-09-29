import gzip
import tempfile
import unittest
from pathlib import Path

from src.ingest.link_graph import build_link_graph
from src.ingest.sql_dump import iter_rows, read_columns, sql_str


def write_dump(directory: Path, table: str, columns: list[str], inserts: list[str]) -> Path:
    """A minimal mysqldump-style file, in the same shape as the real dumps."""
    lines = [f"-- MySQL dump for {table}", f"CREATE TABLE `{table}` ("]
    lines += [f"  `{c}` int(8) NOT NULL," for c in columns]
    lines += [") ENGINE=InnoDB DEFAULT CHARSET=binary;"]
    lines += [f"INSERT INTO `{table}` VALUES {values};" for values in inserts]
    path = directory / f"{table}.sql.gz"
    with gzip.open(path, "wb") as f:
        f.write(("\n".join(lines) + "\n").encode())
    return path


class SqlDumpTest(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.dir = Path(self.tmp.name)

    def tearDown(self):
        self.tmp.cleanup()

    def test_columns_and_tricky_values(self):
        path = write_dump(
            self.dir,
            "page",
            ["page_id", "page_title", "page_random", "page_lang"],
            [
                r"(1,'O\'Brien',0.5,NULL),(2,'A,_(b)_\\',1e-05,'en')",
                r"(3,'',-2,NULL)",
            ],
        )
        self.assertEqual(read_columns(path), ["page_id", "page_title", "page_random", "page_lang"])
        rows = list(iter_rows(path, ("page_id", "page_title", "page_lang")))
        self.assertEqual(
            rows,
            [(b"1", rb"'O\'Brien'", None), (b"2", rb"'A,_(b)_\\'", b"'en'"), (b"3", b"''", None)],
        )
        self.assertEqual(sql_str(rows[0][1]), rb"O\'Brien")

    def test_unparsable_row_raises_instead_of_resynchronizing(self):
        # A hex literal isn't a supported value; the row must not be skipped silently.
        path = write_dump(self.dir, "t", ["a", "b"], ["(1,2),(3,0xFF),(5,6)"])
        with self.assertRaises(ValueError):
            list(iter_rows(path, ("a", "b")))


class BuildLinkGraphTest(unittest.TestCase):
    def test_resolution_and_filtering(self):
        with tempfile.TemporaryDirectory() as tmp:
            d = Path(tmp)
            page = write_dump(
                d,
                "page",
                ["page_id", "page_namespace", "page_title", "page_is_redirect"],
                [
                    "(1,0,'A',0),(2,0,'B',0),(3,0,'C_(film)',0),(4,0,'R',1),(5,0,'R2',1)",
                    r"(6,1,'A',0),(7,0,'O\'Brien',0),(8,0,'Loop',1)",
                ],
            )
            redirect = write_dump(
                d,
                "redirect",
                ["rd_from", "rd_namespace", "rd_title"],
                ["(4,0,'B'),(5,0,'R'),(8,0,'Loop')"],
            )
            linktarget = write_dump(
                d,
                "linktarget",
                ["lt_id", "lt_namespace", "lt_title"],
                [r"(10,0,'B'),(11,0,'R'),(12,0,'R2'),(13,0,'A'),(14,0,'Missing'),(15,0,'O\'Brien'),(16,0,'C_(film)'),(17,1,'A'),(18,0,'Loop')"],
            )
            pagelinks = write_dump(
                d,
                "pagelinks",
                ["pl_from", "pl_from_namespace", "pl_target_id"],
                [
                    "(1,0,10),(1,0,11),(1,0,12)",  # A->B directly, via R, and via R2->R->B: one edge
                    "(1,0,13),(3,0,14),(4,0,10)",  # self-link, red link, link from a redirect
                    "(6,1,13),(3,0,15),(1,0,16)",  # talk-page source; O'Brien; C_(film)
                    "(3,0,17),(3,0,18),(3,0,99)",  # non-mainspace target; redirect loop; unknown target
                ],
            )
            source, target = build_link_graph(page, redirect, linktarget, pagelinks)
        self.assertEqual(list(zip(source.tolist(), target.tolist())), [(1, 2), (1, 3), (3, 7)])


if __name__ == "__main__":
    unittest.main()
