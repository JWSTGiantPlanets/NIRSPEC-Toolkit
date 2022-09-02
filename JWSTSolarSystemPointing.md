# JWST Solar System Pointing

## Purpose

This class will calculate the geometry of an solar system body using the NAIF SPICE kernels, and associated functions. It will accept any body that is available in the NAIF kernels, e.g. planets, moons, comets etc. 

The JWST kernels are available at: [on the ESA website](https://repos.cosmos.esa.int/socci/projects/SPICE_KERNELS/repos/jwst/browse), and the generic NAIF kernels are avilable [on the NAIF website](https://naif.jpl.nasa.gov/naif/data_generic.html).


## Basic usage example

```python
import import JWSTSolarSystemPointing as jssp
import spiceypy as spice
import matplotlib.pyplot as plt


# Load the JWST and Jupiter kernels
kerneldir = '/path/to/kernels/'
spice.furnsh(kerneldir + 'naif0012.tls')
spice.furnsh(kerneldir + 'pck00010.tpc') 
spice.furnsh(kerneldir + 'de430.bsp')
spice.furnsh(kerneldir + 'jup310.bsp')
spice.furnsh(kerneldir + 'jwst_horizons_20211225_20240221_v01.bsp')

# This is a Jupiter JWST NIRSpec IFU cube (as an example)
filename = 'jw01373005001_0310j_00001_nrs1_ifualign_g395h-f290lp_s3d.fits'
geo      = jssp.JWSTSolarSystemPointing(filename)
cube     = geo.full_fov()

# Plot a slice with lat/lon overlaid
fig, ax = plt.subplots()
ax.imshow(geo.im[200, :, :])
ax.contour(geo.get_param('lat'))
ax.contour(geo.get_param('lon'))
plt.show()
```


## Calculated parameters

        lat : degrees
            Planetocsentric latitude
        lon : degrees
            West Longitude
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
            The West longitude of the the point on the equatorial (ring) plane
        ra : degrees
            Right Acension 
        dec : degrees
            Declination