from codecs import latin_1_decode
import numpy as np
from jwst import datamodels
from astropy.io import fits
import spiceypy as spice
from scipy.interpolate import griddata
from astropy import units as u
from astropy import constants as const
import logging

class JWSTSolarSystemPointing: 

    def __init__(self, file, emission_altitude = 0, target = '', arcsec_limit = 0, radec_offset = [0.0, 0.0] ) : 
        '''
        A class that calculates the geometry of JWST Solar System observations

        Parameters
        ----------
        file : string
            The filename of the observation 

        emission_altitude : float
            The altitude above the 1 bar level in km to do the projection at

        target : string
            Specify a target within the frame. The default is set in the datamodels. 
            E.g. when looking at Jupiter data you may want map a moon instead.
            
        arcsec_limit : float
            Limit the geometry calculations withing a radii of the target centre. This can 
            speed things up somewhat.

        radec_offset : float array
            Shift the pixel coordinates in RA and DEC in arcseconds. The JWST pointing is only accurate to ~0.1"
            and so a shift may sometimes be appropriate. 

        '''

        # Configure logging
        logging.basicConfig(level=logging.WARNING)

        self.arcsec_limit = arcsec_limit
        self.radec_offset = radec_offset

        # Load the datamodel and get a copy of the data
        self.hdr = fits.getheader(file, 'PRIMARY')
        model = self.hdr['DATAMODL']
        self.dm = getattr(datamodels, model)(file) 
        self.im = self.dm.data.copy()

        self.observatory = 'JWST'
        if (target) : self.target = target
        else : self.target = self.dm.meta.target.catalog_name

        self.instrument  = self.dm.meta.instrument.name
        self.framestring = 'IAU_' + self.target
        self.iref        = 'J2000'
        self.abcorr      = 'LT'
        
        self.id_obs      = spice.bodn2c(self.observatory)
        self.id_target   = spice.bodn2c(self.target)            
        
        # Determine the mid-point of the observation
        self.obs_start = self.hdr['DATE-BEG']
        self.et_start  = spice.str2et(self.hdr['DATE-BEG'])
        self.et_end    = spice.str2et(self.hdr['DATE-END'])
        self.et        = (self.et_start + self.et_end) / 2.0   

        # Load the appropriate data model
        # Gonna need the right datamodel for each set of data        
        # https://jwst-pipeline.readthedocs.io/en/latest/jwst/datamodels/index.html

        # Define the output names and what are angles
        self.keys   = ['lat', 'lon', 'lat_limb', 'lon_limb', 'lat_graphic', 'phase', 'emission', 'incidence', 'azimuth', 'localtime', 'distance_limb', 'distance_rings', 'lon_rings', 'ra', 'dec']
        self.angles = ['lat', 'lon', 'lat_limb', 'lon_limb', 'lat_graphic', 'phase', 'emission', 'incidence', 'azimuth', 'lon_rings']

        # Create a reciprocal map to the keys
        self.map = {}
        for key, value in enumerate(self.keys) : self.map[value] = key

        self.set_emission_altitude(emission_altitude)
        self.target_location()
        
    def set_emission_altitude(self, emission_altitude) : 
        """Set the altitude of the reference spheroid, relative to IAU 1 bar surface, in km."""        

        self.emission_altitude = emission_altitude
        # Get the radius of the planet + optional altitude offset
        self.radii = spice.bodvar(self.id_target, 'RADII', 3)
        self.radii[0] = self.radii[0] + self.emission_altitude * self.radii[1] / self.radii[2]
        self.radii[1] = self.radii[1] + self.emission_altitude * self.radii[1] / self.radii[2]
        self.radii[2] = self.radii[2] + self.emission_altitude
        self.flattening = ( self.radii[0] - self.radii[2] ) / self.radii[0]     
        
    def target_location(self) : 
        
        # Get the position of the target relative to the obervatory
        self.pos_target, self.light_time = spice.spkpos(self.target, self.et, 
                        self.iref, self.abcorr, self.observatory)
        
        # Convert position to distance, RA, dec
        self.distance, ra, dec = spice.recrad(self.pos_target)
        d, self.lon_obs, self.lat_obs = spice.reclat(self.pos_target)
        self.ra_target = np.rad2deg(ra)
        self.dec_target = np.rad2deg(dec)
        
        # Create the conversion from J2000 to the taret frame
        self.i2p   = spice.pxform(self.iref, self.framestring, self.et - self.light_time)
        self.scloc = np.matmul(-self.i2p, self.pos_target) 
        
    def pixel_params(self, ra, dec) : 
        '''
        lat : degrees
            Planetocsentric latitude
        lon : degrees
            East Longitude (should probs be west) 
        distance_limb : km
            The distance between the pixel and the 1 bar limb. The 1 bar limb is defined as 0 km, 
            and negative distances are on the limb on the planet, positive ones are above the limb.
            Note that, e.g. if you want to project data to a different altitude, use the emission_altitude
            keyword in the initialisation of the gometry object.
        lat_limb : degrees
            Planetocentric latitude of the point on the limb closest to the pixel look vector.
        lon_limb : degrees
            East longitude of the point on the limb closest to the pixel look vector.
        lat_graphic : degrees
            Planetgraphic latitude.
        phase : degrees
            Phase angle
        emissions : degrees
            Emission angle
        incidence : degrees
            Incidence angle
        azimuth : degrees
            Azimuth angle
        localtime : decimal hours
            The localtime of a point
        distance_rings : km
            The distance from the centre of the planet in the equatorial (ring) plane
        lon_rings : degrees
            The East longitude of the the point on the equatorial (ring) plane
        ra : degrees
            Right Acension 
        dec : degrees
            Declination

        '''
        # Set up the return variable
        ret = {}
        for key in self.keys : 
            ret[key] = np.nan
                
        # Get the pixel RA and DEC from the datamodel
        #if (len(self.im.shape) == 2) : coords = self.dm.meta.wcs(x, y)
        #else : coords = self.dm.meta.wcs(x, y, 10)
        
        # If we are only doing a radius around the target
        if (self.arcsec_limit != 0) : 
            dist = np.sqrt((self.ra_target - ra)**2 + (self.dec_target - dec)**2)*3600.0
            if (dist > self.arcsec_limit) : return ret
        
        # Calculate a look vector based on the coordinates and convert to target frame
        vector_J2000  = spice.radrec(self.distance, np.deg2rad(ra), np.deg2rad(dec))
        vector_target = np.matmul(self.i2p, vector_J2000)

        # Get the closest point on the vector to the planet
        origin = np.array([0.0, 0.0, 0.0])
        nearpoint, rayradius = spice.nplnpt(self.scloc, vector_target, origin)

        # Calculate the point in the surface closest to that point
        normal = spice.surfpt(origin, nearpoint, self.radii[0], self.radii[1], self.radii[2])

        # Get the latitude and longitude of the point on the limb
        d, ret['lon_limb'], ret['lat_limb'] = spice.reclat(nearpoint)        
        
        # Calculate the height above the limb 
        ret['distance_limb'] = rayradius - spice.vnorm(normal)    

        # Now get the ring-plane projection
        ringplane = spice.nvc2pl(np.array([0.0, 0.0, 1.0]), 0.0)
        nxpt, ring_intercept = spice.inrypl(self.scloc, vector_target, ringplane)        
        ret['distance_rings'], ret['lon_rings'], lat_rings = spice.reclat(ring_intercept)
        
#        ret['distance_rings'] = spice.vnorm(ring_intercept)
        
        # Test if the pixel vector intersects with our target surface 
        try: 
            point = spice.surfpt(self.scloc, vector_target, self.radii[0], self.radii[1], self.radii[2])
            intercept = True
        except: 
            intercept = False    
    
        if (intercept) : 
        
            # Get some angles 
            ret['phase'], ret['incidence'], ret['emission'] = spice.illum(self.target, self.et , self.abcorr, self.observatory, point)

            # From these angles, calculate the azimut angle (as defined in the NEMESIS manual)
            # https://nemesiscode.github.io/manuals.html
            # Based on zcalc_aziang.pro
            a = np.cos(ret['phase']) - np.cos(ret['emission']) * np.cos(ret['incidence'])
            b = np.sqrt(1.0 - np.cos(ret['emission'])**2) * np.sqrt(1.0 - np.cos(ret['incidence'])**2)            
            ret['azimuth'] = np.pi - np.arccos(a/b)

            # Calculate the planetocentric coordinates 
            distance, ret['lon'], ret['lat'] = spice.reclat(point)
            #ret['distance'] = spice.vnorm(self.scloc - point)
            
             # Calculate the planetographic coordinates
            lon_graphic, ret['lat_graphic'], bodyintercept = spice.recpgr(self.target, point, self.radii[0], self.flattening)
            
            # Get the localtime, and convert to decimal hours 
            hr, min, sc, time, ampm = spice.et2lst(self.et, self.id_target, ret['lon'], 'PLANETOCENTRIC')
            ret['localtime'] = hr + min / 60.0 + sc / 3600.0

        # For the angles, convert radians to degrees
        for key in self.angles : 
            if (ret[key] != np.nan) : ret[key] = np.rad2deg(ret[key])

        # Makes sure longitudes wrap 0 to 360, spice returns the Earth-like -180 to 180. 
        # All longitudes are specifically West! 
        longitudes = ['lon', 'lon_limb', 'lon_rings']
        for key in longitudes : ret[key] = 360 - (ret[key] % 360)

        return ret
    
    def data_to_header(self) : 
        hdr = self.hdr
        for i, key in enumerate(self.keys) : 
            hdr['KEY_' + str(i)] = key
        return hdr
        
    def full_fov(self) : 
        sz = np.flip(self.im.shape)

        valx = np.zeros([sz[1], sz[0]])
        valy = np.zeros([sz[1], sz[0]])
        for x in range(sz[1]) : 
            for y in range(sz[0]) : 
                valx[x, y] = x
                valy[x, y] = y
        
        if (len(self.im.shape) == 2) : 
            coords = self.dm.meta.wcs(valy.flatten(), valx.flatten())
        else : coords = self.dm.meta.wcs(valy.flatten(), valx.flatten(), 10)
        self.ras = (np.reshape(coords[0], (sz[1], sz[0])))
        self.decs = (np.reshape(coords[1], (sz[1], sz[0])))

        # Apply any shift in RA and DEC in arcseconds
        self.ras += self.radec_offset[0] / 3600.0
        self.decs += self.radec_offset[1] / 3600.0

        # Make our output array with extra room for RA and Dec
        output = np.zeros([len(self.keys) + 2, sz[1], sz[0]])
        for x in range(sz[1]) :
            for y in range(sz[0]) :
                ret = self.pixel_params(self.ras[x, y], self.decs[x, y])
                for i, key in enumerate(ret) : 
                    output[i, x, y] = ret[key]

        # Add RA and Dec
        output[-2, : :] = self.ras
        output[-1, : :] = self.decs

        self.geometry_cube = output
        return output
    
    def get_param(self, key) :
        if (key in self.map) : 
            return self.geometry_cube[self.map[key], :, :]
        else : 
            logging.error('Error in get_param(): key "' + str(key) + '" does not exist! Available keys are: ' + ', '.join(self.keys))
            return False


    def get_wavelength(self, xpixel = 0, ypixel = 0) : 
        wave_pixels = np.arange(self.im.shape[0])
        ras, decs, wave = self.dm.meta.wcs(ypixel, xpixel, wave_pixels)
        return wave

    def convert(self, wave, spx_MJysr):
        # Unit converstion from MJy/sr to W/cm2/sr/micron as per Leigh Fletcher
        
        from astropy import units as u

        # Add the correct units
        spx_MJysr = spx_MJysr*u.MJy/u.sr
        
        # Convert "per sr" to "per square arcsecond"
        spx_Jy_per_arcsec= spx_MJysr.to(u.Jy/(u.arcsec*u.arcsec))
        
        # Integrate over solid angle to get irradiance (Jy)
        spx_Jy= spx_Jy_per_arcsec * self.dm.meta.photometry.pixelarea_arcsecsq*u.arcsec*u.arcsec
        # spx_Jy=spx_MJysr * 1e-6 * PIXAR_SR*u.sr

        
        # Now convert Jy to  W/cm2/sr/cm-1
        c=2.99792458e10 #*u.cm/u.s  #cm/s
        corrn=1e-26 #*u.Jy/(u.W/(u.m*u.m)/u.Hz) #(Jy -> W/m2/Hz)
        corrn=corrn/(1.e4)    # W/m2/Hz -> W/cm2/Hz
        corrn=corrn*c       # W/cm2/Hz -> W/cm2/cm-1
        corrn=corrn/self.dm.meta.photometry.pixelarea_steradians #*u.sr    # W/cm2/cm-1 -> W/cm2/sr/cm-1

        spx_Wcm2srcm = spx_Jy*corrn
        
        # Convert  W/cm2/sr/cm-1 to TB

        h=6.626e-34
        c=2.9979e8
        k=1.3806e-23
        spec=spx_Wcm2srcm*1e4/100./u.Jy  # W/cm2/sr/cm-1 -> W/m2/sr/m-1
        v=100.*1e4/wave # wavenumber in m-1
        c1=2*h*c*c
        c2=h*c/k
        a=c1*v*v*v/spec
        TB=c2*v/(np.log(a+1))

        # Now convert back to W/cm2/sr/µm
        l=wave*1e-6
        a=2*h*c*c/(l**5)
        b=np.exp(h*c/(l*k*TB)) - 1
        spx_Wcm2srum = (a/b)/1e4/1e6
        
        #return spx_Jy_per_arcsec, spx_Jy, spx_Wcm2srcm, spx_Wcm2srum, TB
        return spx_Wcm2srum

    def save_spx(self, x, y, wstart = 0, wend = 0, fwhm = 4.0/2600.0, erradd = 0.0) : 
        ''' Create a basic NEMESIS spx input file '''
        wave     = self.get_wavelength(xpixel = x, ypixel = y) 
        spec     = self.convert(wave, self.im[:, y, x] / 3.33000e+08) 
        error    = self.convert(wave, self.dm.err[:, y, x] / 3.33000e+08) 
        lat      = self.geometry_cube[self.map['lat_graphic'], y, x]
        lon      = self.geometry_cube[self.map['lon'], y, x]
        phase    = self.geometry_cube[self.map['phase'], y, x]
        emission = self.geometry_cube[self.map['emission'], y, x]
        azimuth  = self.geometry_cube[self.map['azimuth'], y, x]

        # Add a multipicative error correction
        error += erradd 

        if (wend > 0) : 
            spec  = spec[wave > wstart]
            error = error[wave > wstart]
            wave  = wave[wave > wstart]
            spec  = spec[wave < wend]
            error = error[wave < wend]
            wave  = wave[wave < wend]

        ngeom = 1
        nconv = wave.shape[0]
        nav = 1
        wgeom = 1

        # Generate the header
        header = []
        header.append([fwhm, lat, lon, ngeom])
        header.append([nconv])
        header.append([nav])
        header.append([lat, lon, phase, emission, azimuth, wgeom])

        filename = self.dm.meta.observation.obs_id + '_lon_{:.2f}'.format(lon) + '_lat_{:.2f}.spx'.format(lat)  
        with open(filename, 'w') as f:
            for line in header : 
                f.write('\t'.join('{:.3f}'.format(x) for x in line))
                f.write('\n')
            for i, w in np.ndenumerate(wave) : 
                line = [wave[i], spec[i], error[i]]
                f.write('\t'.join('{:.6e}'.format(x) for x in line))
                f.write('\n')