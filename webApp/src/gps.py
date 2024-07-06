from PIL.ExifTags import TAGS, GPSTAGS
from fractions import Fraction


def get_exif_data(image):
    """
    Retrieve the EXIF data from an image.

    Args:
        image: The image object to extract EXIF data from.

    Returns:
        A dictionary containing the extracted EXIF data.
    """
    exif_data = {}
    info = image._getexif()
    if info:
        for tag, value in info.items():
            tag_name = TAGS.get(tag, tag)
            exif_data[tag_name] = value
    return exif_data


def get_gps_info(exif_data):
    """
    Extracts GPS information from the given EXIF data.

    Args:
        exif_data (dict): The EXIF data containing GPS information.

    Returns:
        dict: A dictionary containing the extracted GPS information.
    """
    gps_info = {}
    for key in exif_data.get('GPSInfo', {}).keys():
        decode = GPSTAGS.get(key, key)
        gps_info[decode] = exif_data['GPSInfo'][key]
    return gps_info


def convert_to_degrees(value):
    """
    Converts a coordinate value from degrees, minutes, and seconds format to decimal degrees format.
    
    Args:
        value (tuple): A tuple containing the degrees, minutes, and seconds values of a coordinate.
        
    Returns:
        float: The converted coordinate value in decimal degrees format.
    """
    def convert_fraction(value):
        return float(value.num) / float(value.den) if isinstance(value, Fraction) else float(value)
    
    d = convert_fraction(value[0])
    m = convert_fraction(value[1])
    s = convert_fraction(value[2])
    return d + (m / 60.0) + (s / 3600.0)


def get_lat_lon(gps_info):
    """
    Get the latitude and longitude from GPS information.

    Args:
        gps_info (dict): A dictionary containing GPS information.

    Returns:
        tuple: A tuple containing the latitude and longitude. If the GPS information is incomplete, returns (None, None).
    """
    if 'GPSLatitude' in gps_info and 'GPSLongitude' in gps_info:
        lat = convert_to_degrees(gps_info['GPSLatitude'])
        if gps_info['GPSLatitudeRef'] != 'N':
            lat = -lat

        lon = convert_to_degrees(gps_info['GPSLongitude'])
        if gps_info['GPSLongitudeRef'] != 'E':
            lon = -lon

        return lat, lon
    else:
        return None, None


def extraeLatitudLongitud(imagen):
    """
    Extracts latitude and longitude information from the given image.

    Parameters:
    imagen (str): The path to the image file.

    Returns:
    tuple: A tuple containing the latitude and longitude values extracted from the image.
           If the image does not contain GPS information, (None, None) is returned.
    """
    exif_data = get_exif_data(imagen)
    if 'GPSInfo' in exif_data:
        gps_info = get_gps_info(exif_data)
        return get_lat_lon(gps_info)
    else:
        return None, None