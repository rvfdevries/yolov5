from PIL import Image
from PIL.ExifTags import TAGS, GPSTAGS
from datetime import datetime

def _get_if_exist(data, key):
    if key in data:
        return data[key]
		
    return None

def _convert_to_degrees(value):
    """Helper function to convert the GPS coordinates stored in the EXIF to degress in float format"""
#    d = value[0]
#    m = value[1]
#    s = value[2]
#
#    return d + (m / 60.0) + (s / 3600.0)

    d0 = value[0][0]
    d1 = value[0][1]
    d  = float(d0) / float(d1)

    m0 = value[1][0]
    m1 = value[1][1]
    m  = float(m0)/float(m1)

    s0 = value[2][0]
    s1 = value[2][1]
    s  = float(s0)/float(s1)

    return d + (m / 60.0) + (s / 3600.0)


def get_exif_data(image):
#    """Returns a dictionary from the exif data of an PIL Image item. Also converts the GPS Tags"""
    exif_data = {}

    info = image._getexif() 
    exif_data = info
    
    if info:
#        for tag, value in info.items():
        for tag, value in list(info.items()):
            decoded = TAGS.get(tag, tag)
            if decoded == "GPSInfo":
                gps_data = {}
                for t in value:
                    sub_decoded = GPSTAGS.get(t, t)
                    gps_data[sub_decoded] = value[t]
    
                exif_data[decoded] = gps_data
            else:
                exif_data[decoded] = value
    
    return exif_data
#
    
def get_lat_lon(exif_data):
    """Returns the latitude and longitude, if available, from the provided exif_data (obtained through get_exif_data above)"""
    lat = None
    lon = None

    if "GPSInfo" in exif_data:		
        gps_info = exif_data["GPSInfo"]

        gps_latitude        = _get_if_exist(gps_info, "GPSLatitude")
        gps_latitude_ref    = _get_if_exist(gps_info, 'GPSLatitudeRef')
        gps_longitude       = _get_if_exist(gps_info, 'GPSLongitude')
        gps_longitude_ref   = _get_if_exist(gps_info, 'GPSLongitudeRef')
        gps_timestamp       = _get_if_exist(gps_info, 'GPSTimeStamp')
        gps_date            = _get_if_exist(gps_info, 'GPSDateStamp')
        
#        print(gps_timestamp)

        hour                = int(gps_timestamp[0][0])
        minute              = int(gps_timestamp[1][0])
        second              = int(gps_timestamp[2][0])

#        hour                = int(gps_timestamp[0])
#        minute              = int(gps_timestamp[1])
#        second              = int(gps_timestamp[2])
        
        year                = int(gps_date[:4])
        month               = int(gps_date[5:7])
        day                 = int(gps_date[8:10])
        
#        print(hour)
#        print(minute)
#        print(second)
#        print(year)
#        print(month)
#        print(day)

        if gps_latitude and gps_latitude_ref and gps_longitude and gps_longitude_ref:
            lat = _convert_to_degrees(gps_latitude)
            if gps_latitude_ref != "N":                     
                lat = 0 - lat

            lon = _convert_to_degrees(gps_longitude)
            if gps_longitude_ref != "E":
                lon = 0 - lon

        dt                  = datetime(year = year, month = month, day = day, hour = hour, minute = minute, second = second)

        return lat, lon, dt


def gopro_get_GPS(path):
    
    img                 = Image.open(path)    

    info                = get_exif_data(img)
    
    if info != None:
        latlon              = get_lat_lon(info)
        
        if latlon != None:
            
    #        print(latlon)
            
            # convert to EPSG:32606 (from EPSG:4326) [this is WGS 84 --> WGS 84 / UTM Zone 6N]
            lon                 = latlon[1]
            lat                 = latlon[0]    
            dt                  = latlon[2]
                            
            return lat, lon, dt
