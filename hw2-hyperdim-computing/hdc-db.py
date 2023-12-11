import csv
from typing import Any, Optional

import numpy as np

from hdc import *


class HDDatabase:

    def __init__(self):
        self.db = HDItemMem("db")
        """Contains bundled together entries for each field of each row, keyed by primary key
        Each entry is a binding of the field name and the field value"""

        # other instantiations here
        self.string_cb = HDCodebook(name='string_cb')
        "Contains primary key strings"

        self.field_cb = HDCodebook(name='field_cb')
        "Contains all field names in the database"

        self.value_cb = HDCodebook(name='value_cb')
        "Contains all possible values in the database"

    def encode_string(self, value: str) -> np.ndarray:
        """translate a string to a hypervector"""
        if self.string_cb.has(value):
            return self.string_cb.get(value)
        else:
            self.string_cb.add(value)
            return self.string_cb.get(value)
        # return make_word(self.letter_cb, value)

    def decode_string(self, hypervec: np.ndarray) -> str:
        """translate a hypervector to a string"""
        key, dist = self.string_cb.wta(hypervec)
        return key

    def encode_row(self, fields: dict[Any, Any]):
        """translate a dictionary of field-value pairs to a hypervector"""
        for k, v in fields.items():
            if not self.field_cb.has(k):
                self.field_cb.add(k)
            if not self.value_cb.has(v):
                self.value_cb.add(v)

        # We bind together the field name and field value hypervectors, and create a bundle of all of these bindings
        return HDC.bundle([
            HDC.bind(self.field_cb.get(k), self.value_cb.get(v))
            for k, v in fields.items()
        ])

    def _decode_field_from_row(self, hypervec: np.ndarray, field_name: str) -> Optional[tuple[Any, float]]:
        # Bind the hypervector with the field name hypervector
        cleaned_binding = HDC.bind(hypervec, self.field_cb.get(field_name))
        # Check if the result has any value in its matches
        if m := self.value_cb.matches(cleaned_binding):
            return min(m.keys(), key=m.get), min(m.values())
        return None

    def decode_row(self, hypervec: np.ndarray) -> dict[Any, Any]:
        """reconstruct a dictionary of field-value pairs from a hypervector."""
        result_dict = {}
        for field_name in self.field_cb.all_keys():
            if (m := self._decode_field_from_row(hypervec, field_name)) is not None:
                result_dict[field_name] = m[0]
        return result_dict

    def add_row(self, primary_key: str, fields: dict):
        row_hv = self.encode_row(fields)
        self.db.add(primary_key, row_hv)

    def get_row(self, row: str) -> dict:
        """retrieve a dictionary of field-value pairs from a hypervector row"""
        return self.decode_row(self.db.get(row))

    def get_value(self, key: str, field: str):
        """given a primary key and a field, get the value assigned to the field"""
        row_hypervec = self.db.get(key)
        field_value = self._decode_field_from_row(row_hypervec, field)[0]
        return field_value

    def get_matches(self, field_value_dict, threshold=0.4):
        """get database entries that contain provided dictionary of field-value pairs"""
        query_hv = self.encode_row(field_value_dict)
        return self.db.matches(query_hv, threshold=threshold)

    def get_analogy(self, target_key: str, other_key: str, target_value: Any) -> Optional[tuple[Any, float]]:
        """analogy query"""
        target_hypervec = self.db.get(target_key)
        cleaned_target_hypervec = HDC.bind(target_hypervec, self.value_cb.get(target_value))
        key, dist = self.field_cb.wta(cleaned_target_hypervec)
        decoded_field: Optional[tuple[Any, float]] = self._decode_field_from_row(self.db.get(other_key), field_name=key)
        return decoded_field


def load_json():
    data = {}
    with open("digimon.csv", "r") as csvf:
        csvReader = csv.DictReader(csvf)
        for rows in csvReader:
            key = rows['Digimon']
            data[key] = rows

    return data


def build_database(data):
    HDC.SIZE = 10000
    db = HDDatabase()

    for key, fields in data.items():
        db.add_row(key, fields)

    return db


def summarize_result(data, result, summary_fn):
    print("---- # matches = %d ----" % len(list(result.keys())))
    for digi, distance in result.items():
        print("%f] %s: %s" % (distance, digi, summary_fn(data[digi])))


def digimon_basic_queries(data, db):
    print("===== virus-plant query =====")
    thr = 0.40
    digis = db.get_matches({"Type": "Virus", "Attribute": "Plant"}, threshold=thr)
    summarize_result(data, digis, lambda row: "true match" if row["Type"] == "Virus" and row[
        "Attribute"] == "Plant" else "false positive")

    print("===== champion query =====")
    thr = 0.40
    digis = db.get_matches({"Stage": "Champion"}, threshold=thr)
    summarize_result(data, digis, lambda row: "true match" if row["Stage"] == "Champion" else "false positive")


def digimon_test_encoding(data, db):
    strn = "tester"
    hv_test = db.encode_string(strn)
    rec_strn = db.decode_string(hv_test)
    print("original=%s" % strn)
    print("recovered=%s" % rec_strn)
    print("---")

    row = data["Wormmon"]
    hvect = db.encode_row(row)
    rec_row = db.decode_row(hvect)
    print("original=%s" % str(row))
    print("recovered=%s" % str(row))
    print("---")


def digimon_value_queries(data, db):
    value = db.get_value("Lotosmon", "Stage")
    print("Lotosmon.Stage = %s" % value)

    targ_row = db.get_row("Lotosmon")
    print("Lotosmon" + str(targ_row))


def analogy_query(data, db):
    # Lotosmon is to Data as Crusadermon is to <what field>

    targ_row = db.get_row("Lotosmon")
    other_row = db.get_row("Crusadermon")
    print("Lotosmon has a a field with a Data value, what is the equivalent value in Crusadermon's entry")
    value, dist = db.get_analogy(target_key="Lotosmon", other_key="Crusadermon", target_value="Data")

    print("Lotosmon" + str(targ_row))
    print("Crusadermon" + str(other_row))
    print("------")
    print("value: %s (dist=%f)" % (value, dist))
    print("expected result: Virus, the type of Crusadermon")
    print("")


def __main__():
    data = load_json()
    db = build_database(data)
    digimon_basic_queries(data, db)
    digimon_value_queries(data, db)
    digimon_test_encoding(data, db)
    analogy_query(data, db)


if __name__ == '__main__':
    __main__()
