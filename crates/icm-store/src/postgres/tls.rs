//! TLS for the PostgreSQL connection.
//!
//! Follows the connection string's `sslmode`: `disable` never uses TLS, and
//! the default (`prefer`) tries TLS and falls back to plain text, as libpq
//! does. `require` fails when the server offers no TLS. When TLS is used the
//! server certificate is always verified, against the system roots plus the
//! PEM file in `ICM_POSTGRES_CA_FILE` if set (for a private or provider CA).

use std::sync::Arc;

use rustls::pki_types::{CertificateDer, pem::PemObject};
use rustls::{ClientConfig, RootCertStore};
use tokio_postgres_rustls::MakeRustlsConnect;

use super::*;

pub(crate) const CA_FILE_ENV: &str = "ICM_POSTGRES_CA_FILE";

fn tls_err(what: &str, e: impl std::fmt::Display) -> IcmError {
    IcmError::Config(format!("PostgreSQL TLS: {what}: {e}"))
}

/// Build the TLS connector for a connection string.
pub(crate) fn connector(url: &str) -> IcmResult<MakeRustlsConnect> {
    let mut roots = RootCertStore::empty();
    if !url.contains("sslmode=disable") {
        roots.add_parsable_certificates(rustls_native_certs::load_native_certs().certs);
        if let Ok(path) = std::env::var(CA_FILE_ENV) {
            for cert in CertificateDer::pem_file_iter(&path)
                .map_err(|e| tls_err(&format!("cannot read {CA_FILE_ENV} {path}"), e))?
            {
                let cert = cert.map_err(|e| tls_err(&format!("bad certificate in {path}"), e))?;
                roots
                    .add(cert)
                    .map_err(|e| tls_err(&format!("unusable certificate in {path}"), e))?;
            }
        }
    }
    let config =
        ClientConfig::builder_with_provider(Arc::new(rustls::crypto::ring::default_provider()))
            .with_safe_default_protocol_versions()
            .map_err(|e| tls_err("protocol versions", e))?
            .with_root_certificates(roots)
            .with_no_client_auth();
    Ok(MakeRustlsConnect::new(config))
}
