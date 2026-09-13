ENV["DATADEPS_ALWAYS_ACCEPT"] = "true"

using DataDeps

register(
    DataDep(
        "bccd16",
        "BCCD16 is an invalid correlation matrix of dimension 3250 constructed from data for 3250 banks in 27 EU member states (EU 27).",
        "https://github.com/higham/matrices-correlation-invalid/raw/refs/heads/master/bccd16.mat",
        "da84ad3a249b3857d320151901f9093b50ff010f84dd897a97ef1de94f483c78"
    )
)
register(
    DataDep(
        "cor1399",
        "COR1399 is an invalid correlation matrix of dimension 1399 constructed from stock data.  The matrix was provided by investment company Orbis",
        "https://github.com/higham/matrices-correlation-invalid/raw/refs/heads/master/cor1399.mat",
        "b20c1ce88d6432189b7559d24d3ae849ca48059e71e87fbc96004c7a7b2ca3eb"
    )
)
register(
    DataDep(
        "cor3120",
        "COR3120 is an invalid correlation matrix of dimension 3120 constructed from stock data.  The matrix was provided by investment company Orbis.",
        "https://github.com/higham/matrices-correlation-invalid/raw/refs/heads/master/cor3120.mat",
        "5ab0ce21d68e216b594c041f0ccd01e43e9e3624ea9eb5d31623bd52b6eae032"
    )
)
register(
    DataDep(
        "usgs13",
        """
        USGS13 is a matrix is for carbon dioxide storage assessment units for the Rocky
        Mountains region of the USA and was generated during the
        national assessment of carbon dioxide storage resources.

        Source:
        U.S. Geological Survey Geologic Carbon Dioxide Storage
        Resources Assessment Team. National Assessment of Geologic Carbon
        Dioxide Storage Resources---Results (Ver. 1.1, September 2013),
        September 2013. Provided by Madalyn Blondes of the U.S. Geological
        Survey, email correspondence; permission to use given.
        """,
        "https://github.com/higham/matrices-correlation-invalid/raw/refs/heads/master/Rocky_Mountain_Region_CORR.mat",
        "5c89504736d33723be24189f6e427552b5c913455e73bb26c7a2108865d1509d"
    )
)
