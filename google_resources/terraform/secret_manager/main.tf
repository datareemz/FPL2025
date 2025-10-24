resource "google_secret_manager_secret" "fpl_credentials" {
  project   = var.project_id
  secret_id = var.secret_id

  replication {
    auto {}
  }
}

resource "google_secret_manager_secret_version" "fpl_credentials_version" {
  secret      = google_secret_manager_secret.fpl_credentials.id
  secret_data = var.secret_data
}