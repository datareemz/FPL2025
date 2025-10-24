resource "google_storage_bucket" "fpl_bucket" {
  name          = var.bucket_name
  location      = upper(var.region)
  storage_class = var.storage_class
  project       = var.project_id

  versioning {
    enabled = var.versioning_enabled
  }

  lifecycle_rule {
    condition {
      age = var.lifecycle_age_days
    }
    action {
      type = "Delete"
    }
  }

  uniform_bucket_level_access = true
}