from PIL.ExifTags import TAGS, GPSTAGS
from fractions import Fraction

def get_exif_data(image):
    """Devuelve un diccionario con los datos EXIF de la imagen."""
    exif_data = {}
    info = image._getexif()
    if info:
        for tag, value in info.items():
            tag_name = TAGS.get(tag, tag)
            exif_data[tag_name] = value
    return exif_data

def get_gps_info(exif_data):
    """Extrae la información GPS de los datos EXIF."""
    gps_info = {}
    for key in exif_data.get('GPSInfo', {}).keys():
        decode = GPSTAGS.get(key, key)
        gps_info[decode] = exif_data['GPSInfo'][key]
    return gps_info

def convert_to_degrees(value):
    """Convierte las coordenadas GPS en grados, minutos y segundos a formato decimal."""
    def convert_fraction(value):
        return float(value.num) / float(value.den) if isinstance(value, Fraction) else float(value)
    
    d = convert_fraction(value[0])
    m = convert_fraction(value[1])
    s = convert_fraction(value[2])
    return d + (m / 60.0) + (s / 3600.0)

def get_lat_lon(gps_info):
    """Devuelve la latitud y longitud en formato decimal."""
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
    # Extraer la información GPS
    exif_data = get_exif_data(imagen)
    if 'GPSInfo' in exif_data:
        gps_info = get_gps_info(exif_data)
        return get_lat_lon(gps_info)
    else:
        return None, None