from typing import Any
from json import JSONEncoder, JSONDecoder
import numpy as np

from ..colormap import ColorMap
from ..Qt.QtGui import QColor

"""
This class provides custom JSON encoder and decoder for objects that are not supported by the standard json library.

Parameter objects may contains several data types of objects that cannot be directly serialized into JSON.
To ensure full serialization and deserialization of Parameter objects, this class introduced custom encoding for:
	- Numpy ndarray: Not serializable in JSON by default.
	- ColorMap: Not serializable in JSON by default.
	- tuple: the json module encodes tuples as lists, so a custom encoding is required to preserve and restore the tuple
	 type.

We choose to handle JSON encoding in a dedicated class rather than directly inside the Parameter classes. Indeed, many
Parameter attributes can accept multiple data types, making direct handling tricky. A dedicated encoder ensures a clean
and maintainable encoding for all Parameter classes.
"""


class JsonEncoderDecoder(JSONEncoder):
    def default(self, o: Any) -> Any:
        """
        Serialize additional objects for JSON encoding.
        Override the default() method from JSONEncoder to support:
            - NumPy ndarray,
            - ColorMap.
        Returns a serializable object for o, or call the base implementation.

        Parameters
        ----------
        o: Any
            The object to serialize.

        Returns
        -------
        Any
            A serializable representation of the object.
        """
        if isinstance(o, np.ndarray):
            return {'__ndarray__': o.tolist()}

        elif isinstance(o, ColorMap):
            attrs = dict()
            attrs['pos'], attrs['color'] = o.getStops()
            attrs['mapping_mode'] = o.mapping_mode
            attrs['name'] = o.name

            return {'__colormap__': attrs}

        return super().default(o)

    def encode(self, o: Any) -> str:
        """
        Encode an object into a JSON string.
        Extends encode() method from JSONEncoder to support tuples.

        Parameters
        ----------
        o: Any
            The object to encode.

        Returns
        -------
        str
            A JSON string representation of the object.
        """

        def hint_special(o: Any):
            if isinstance(o, tuple):
                return {'__tuple__': [hint_special(e) for e in o]}
            elif isinstance(o, list):
                return [hint_special(e) for e in o]
            elif isinstance(o, dict):
                return {k: hint_special(v) for k, v in o.items()}
            else:
                return o

        return super(JsonEncoderDecoder, self).encode(hint_special(o))

    @staticmethod
    def _decode_hook(dct: dict) -> np.ndarray | ColorMap | tuple | dict:
        """
        Decoding hook used for JSON deserialization.
        Recognizes and reconstructs:
            - NumPy ndarray,
            - ColorMap,
            - tuple.

        Parameters
        ----------
        dct: dict
            The decoded dictionary.

        Returns
        -------
        np.ndarray or ColorMap or tuple or dict
            The deserialized object.
        """
        if '__ndarray__' in dct:
            return np.array(dct['__ndarray__'])

        if '__colormap__' in dct:
            elt = dct['__colormap__']
            return ColorMap(pos=elt['pos'], color=elt['color'], mapping=elt['mapping_mode'], name=elt['name'])

        if '__tuple__' in dct:
            return tuple(dct['__tuple__'])

        return dct

    def decode(self, json_str: str) -> dict:
        """
        Decode a JSON string and recreate the corresponding object.

        Parameters
        ----------
        json_str: str
            The JSON string to decode.

        Returns
        -------
        dict
            The decoded object.
        """
        return JSONDecoder(object_hook=JsonEncoderDecoder._decode_hook).decode(json_str)