# Create the Cloud Run Job.
# defines the containerized task that will be executed.
resource "google_cloud_run_v2_job" "python_job" {
  project  = var.project_id
  location = var.region
  name     = var.job_name
  deletion_protection = false
  template {
    template {
      containers {
        image = var.container_image
      }
    }
  }
  depends_on = [
    var.api_services_dependency
  ]
}