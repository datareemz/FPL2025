variable "project_id" {
  type        = string
  description = "Your Google Cloud project ID."
}

variable "region" {
  type        = string
  description = "The GCP region (e.g., 'us-central1')."
}

variable "cloud_run_job_name" {
  type        = string
  description = "The name of the Cloud Run job to schedule."
}

variable "cloud_run_job_id" {
  type        = string
  description = "The full ID of the Cloud Run job to schedule."
}

variable "cron_schedule" {
  type        = string
  description = "The cron schedule for the job"
  default = "5 4 * * *"
}

variable "scheduler_service_account" {
  type        = string
  description = "Email of the scheduler service account"
}