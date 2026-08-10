/** @type {import('next').NextConfig} */
const nextConfig = {
  // Produit une image Docker minimale ne contenant que le serveur Next requis.
  output: "standalone",
  reactCompiler: true,
  // En développement, cette réécriture conserve une origine unique et évite
  // CORS. En Docker, le proxy Nginx envoie directement /api au service Rust.
  async rewrites() {
    const backend = process.env.VIDIOAI_BACKEND_URL || "http://127.0.0.1:8080";
    return [
      { source: "/api/:path*", destination: `${backend}/api/:path*` },
      { source: "/healthcheck", destination: `${backend}/healthcheck` },
    ];
  },
};

export default nextConfig;
