import os
import urllib.request
import zipfile
import tempfile
import geopandas as gpd

def main():
    print("NYC TLC Taxi Zones Shapefile indiriliyor...")
    url = "https://d37ci6vzurychx.cloudfront.net/misc/taxi_zones.zip"
    
    script_dir = os.path.dirname(os.path.abspath(__file__))
    public_dir = os.path.abspath(os.path.join(script_dir, "..", "web", "public"))
    geojson_path = os.path.join(public_dir, "nyc_taxi_zones.geojson")
    
    with tempfile.TemporaryDirectory() as tmpdir:
        zip_path = os.path.join(tmpdir, "taxi_zones.zip")
        
        import requests
        import time
        max_retries = 5
        success = False
        for attempt in range(1, max_retries + 1):
            try:
                print(f"Indirme denemesi {attempt}/{max_retries}...")
                r = requests.get(url, stream=True, verify=False, timeout=60)
                r.raise_for_status()
                with open(zip_path, 'wb') as f:
                    for chunk in r.iter_content(chunk_size=8192):
                        f.write(chunk)
                success = True
                print("Indirme tamamlandi.")
                break
            except Exception as e:
                print(f"Hata: {e}")
                time.sleep(3)
                
        if not success:
            raise RuntimeError("Veri indirilemedi.")
                
        print("Zip cikariliyor...")
        with zipfile.ZipFile(zip_path, 'r') as zip_ref:
            zip_ref.extractall(tmpdir)
        
        # .shp dosyasini dinamik olarak bul
        shp_path = None
        for root, dirs, files in os.walk(tmpdir):
            for file in files:
                if file.endswith(".shp"):
                    shp_path = os.path.join(root, file)
                    break
            if shp_path:
                break
                
        if not shp_path:
            raise RuntimeError("Zip icerisinde .shp dosyasi bulunamadi.")
            
        print(f"Shapefile okunuyor (geopandas ile): {shp_path}")
        gdf = gpd.read_file(shp_path)
        
        print("WGS84 (EPSG:4326) projeksiyonuna cevriliyor...")
        gdf = gdf.to_crs(epsg=4326)
        
        print(f"GeoJSON formatinda kaydediliyor: {geojson_path}")
        gdf.to_file(geojson_path, driver="GeoJSON")
        print("Basariyla tamamlandi!")

if __name__ == "__main__":
    main()
