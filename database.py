# -*- coding: utf-8 -*-

#------------------------------------------------------------------------------
#
# Dataset hanndling such as ImageCollection and their Annotations
#
#------------------------------------------------------------------------------

import sqlite3
import dateutil
import time
import os


class Database:
    url = ''
    connection = 0

    def __init__(self, url):
        self.url = url
        if len(url) > 0:
            self.connect()

    def connect(self):
        # Make connection
        self.connection = sqlite3.connect(self.url)

    def close(self):
        self.connection.close()

    def initialize_flickr_demo_tables(self):
        # Create table for images
        self.connection.execute('''CREATE TABLE IF NOT EXISTS Images
                                    (img_id INTEGER PRIMARY KEY AUTOINCREMENT,
                                    img_name TEXT,
                                    time_taken INTEGER,
                                    time_uploaded INTEGER,
                                    latitude REAL,
                                    longitude REAL)''')
        # Commit changes
        self.connection.commit()

        # Bag-of-feature histogram distance matrix table
        self.connection.execute('''CREATE TABLE IF NOT EXISTS BOF_distances
                                    (img_id1 INTEGER,
                                    img_id2 INTEGER,
                                    distance REAL)''')
        self.connection.commit()

        # RGB color histogram distances
        self.connection.execute('''CREATE TABLE IF NOT EXISTS RGB_distances
                                    (img_id1 INTEGER,
                                    img_id2 INTEGER,
                                    distance REAL)''')
        self.connection.commit()

        # Keywords e.g. tags
        self.connection.execute('''CREATE TABLE IF NOT EXISTS Keywords
                                    (key_id INTEGER PRIMARY KEY AUTOINCREMENT,
                                    keyword TEXT)''')

        self.connection.commit()
        self.connection.execute('''CREATE TABLE IF NOT EXISTS img2keyword
                                    (img_id INTEGER,
                                    key_id INTEGER)''')
        self.connection.commit()

    def add_image(self, imageName):

        # TODO: get these numbers somewhere
        time_taken = 0
        time_uploaded = 0

        self.connection.execute("""INSERT INTO Images
                                (img_name, time_taken, time_uploaded)
                                VALUES (?, ?, ?)""",
                                (imageName, time_taken, time_uploaded))

        self.connection.commit()

    def set_bof_distance(self, img_id1, img_id2, distance):
        # Check that inputs are integers
        img_id1 = int(img_id1)
        img_id2 = int(img_id2)
        # Make sure that img_id1 < img_id2 so that our db table is correct
        if img_id1 > img_id2:
            tmpid = img_id1
            img_id1 = img_id2
            img_id2 = tmpid

        # Get row if it exist already to updated it
        d = self.get_bof_distance(img_id1, img_id2)

        # If false (no record yet) insert a new one
        if d == False:
            self.connection.execute('''INSERT INTO BOF_distances
                                       (img_id1, img_id2, distance)
                                       VALUES (?, ?, ?)''',
                                       (img_id1, img_id2, distance))
            self.connection.commit()
        else:
            self.connection.execute('''UPDATE BOF_distances
                                       SET distance=:distance
                                       WHERE img_id1=:img_id1 AND
                                             img_id2=:img_id2''',
                                       {'img_id1': img_id1, 'img_id2': img_id2,
                                       'distance': distance})
            self.connection.commit()

    def get_bof_distance(self, img_id1, img_id2):
        query = self.connection.execute('''SELECT distance FROM BOF_distances
                                   WHERE img_id1=:img_id1 AND img_id2=:img_id2
                                   OR img_id1=:img_id2 AND img_id2=:img_id1''',
                                   {'img_id1': img_id1, 'img_id2': img_id2})
        result = query.fetchone()
        if result == None:
            return False
        else:
            return result

    # Color histogram distance functions
    def set_rgb_distance(self, img_id1, img_id2, distance):
        # Check that inputs are integers
        img_id1 = int(img_id1)
        img_id2 = int(img_id2)
        # Make sure that img_id1 < img_id2 so that our db table is correct
        if img_id1 > img_id2:
            tmpid = img_id1
            img_id1 = img_id2
            img_id2 = tmpid

        # Get row if it exist already to updated it
        d = self.get_rgb_distance(img_id1, img_id2)

        # If false (no record yet) insert a new one
        if d == False:
            self.connection.execute('''INSERT INTO RGB_distances
                                       (img_id1, img_id2, distance)
                                       VALUES (?, ?, ?)''',
                                       (img_id1, img_id2, distance))
            self.connection.commit()
        else:
            self.connection.execute('''UPDATE RGB_distances
                                       SET distance=:distance
                                       WHERE img_id1=:img_id1 AND
                                             img_id2=:img_id2''',
                                       {'img_id1': img_id1, 'img_id2': img_id2,
                                       'distance': distance})
            self.connection.commit()

    def get_rgb_distance(self, img_id1, img_id2):
        query = self.connection.execute('''SELECT distance FROM RGB_distances
                                   WHERE img_id1=:img_id1 AND img_id2=:img_id2
                                   OR img_id1=:img_id2 AND img_id2=:img_id1''',
                                   {'img_id1': img_id1, 'img_id2': img_id2})
        result = query.fetchone()
        if result == None:
            return False
        else:
            return result

    def del_image(self, imageName):
        self.connection.execute("""DELETE FROM Images
                                    WHERE img_name=:imgname""",
                                    {"imgname", str(imageName)})

    def get_image_id(self, imageName):
        query = self.connection.execute('''SELECT img_id FROM Images
                                           WHERE img_name=:imgname''',
                                           {'imgname': imageName})
        results = query.fetchone()

        return results[0]

    def get_image(self, imageName):
        query = self.connection.execute('''SELECT * FROM Images
                                    WHERE img_name=:imgname''',
                                    {'imgname': imageName})
        row = query.fetchone()

        return row

    def get_imagelist(self):
        query = self.connection.execute('''SELECT img_name FROM Images''')
        result = query.fetchall()
        return result

    def set_image_metadata(self, metadata, imageName='', img_id=0):
        # If user did not give metadata, we can quit
        if metadata == None:
            return

        # If user gave name of the image instead of id we need to fetch
        if img_id == 0:
            img_id = self.get_image_id(imageName)

        timetaken = 0

        # Convert time string into int timestamp
        if metadata.has_key('CreateDate'):
            timetakenstr = metadata['CreateDate']
            try:
                dt = dateutil.parser.parse(timetakenstr)
                timetaken = int(time.mktime(dt.timetuple()))
            except:
                print "Could not parse: " +  timetakenstr + " for image " + imageName
        else:
            timetaken = 0

        if metadata.has_key('flickr_longitude'):
            longitude = metadata['flickr_longitude']
        else:
            longitude = 0
        if metadata.has_key('flickr_latitude'):
            latitude = metadata['flickr_latitude']
        else:
            latitude = 0

        # Make query
        self.connection.execute('''UPDATE Images
                                   SET longitude=:longitude,
                                       latitude=:latitude,
                                       time_taken=:timetaken
                                    WHERE img_id=:imgid''',
                                    {'longitude': longitude,
                                     'latitude': latitude,
                                     'timetaken': timetaken,
                                     'imgid': img_id})
        self.connection.commit()

    def add_image_to_cluster(self, imgFiles, C, d):

        for i in range(len(imgFiles)):
            imgFilename = imgFiles[i][0][0]
            cluster_id = C[i][0]
            dist = d[i][0]

            print imgFilename + '\t' + str(cluster_id) + '\t' + str(dist)

            # Get id of the image
            self.connection.execute("""SELECT img_id, img_name
                                        FROM images
                                        WHERE img_name=:imgname""",
                                        {"imgname": imgFilename})
            row = self.connection.fetchone()

            if row != None:
                imgId = row[0]

                # Check if the image is already in the cluster
                self.connection.execute("""SELECT img_id, cluster_id
                                            FROM img2cluster
                                            WHERE img_id=:imgid""", {"imgid": imgId})
                row = self.connection.fetchone()

                if row == None:
                    # Prepare insert statement
                    self.connection.execute("""INSERT INTO img2cluster
                                                (cluster_id, img_id, distance)
                                                VALUES (?,?,?)""",
                                                (int(cluster_id), int(imgId), float(dist)))

                    # Commit insert
                    self.connection.commit()
                else:
                    # Image was already in a cluster, lets update its cluster
                    # to be sure that the cluster is correct

                    # Check if we have something to be updated
                    if row[0] != int(cluster_id):
                        self.connection.execute("""UPDATE img2cluster
                                    SET cluster_id=?, distance=?
                                    WHERE img_id=?""",
                                    (int(cluster_id), int(dist), int(imgId)))
                        self.connection.commit()

def metadata_read_flickr_datafile(filename):

    if os.path.exists(filename):

        fp = file(filename)
        # Initialize metadata dictionary
        metadata = {}
        # Read each line and fill info that can be usefull
        line = fp.readline()

        while len(line) > 0:
            #print line.split(';')
            # Parse line into key and value which is so simple in python!
            splits = line.split(';')
            key = splits[0]
            value = ';'.join(splits[1:]) #splits[1:].join
            # Store it into Dictinary
            metadata[key] = value[0:-1]  # Last item is end of line which we dont need
            # Read new line
            line = fp.readline()

        fp.close()

        return metadata
    else:
        return None