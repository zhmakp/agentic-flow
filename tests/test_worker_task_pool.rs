use std::{
    sync::{
        Arc,
        atomic::{AtomicUsize, Ordering},
    },
    time::Duration,
};

use agentic_flow_lib::worker::TaskPool;
use tokio::{sync::Mutex, time::sleep};

/// Test creating a new task pool with default configuration
#[tokio::test]
async fn test_new_task_pool() {
    let print_task_processing = Arc::new(Mutex::new(|task: i32| {
        println!("Processing task: {:?}", task);
    }));
    let pool = TaskPool::<i32>::new(2, print_task_processing).await;
    assert_eq!(pool.worker_count(), 2);
    assert_eq!(pool.capacity(), 100);

    // Clean shutdown
    pool.shutdown().await;
}

/// Test executing a task in the TaskPool
#[tokio::test]
async fn test_taskpool_execute_task() {
    // Use an atomic counter to verify execution
    let counter = Arc::new(AtomicUsize::new(0));
    let counter_clone = counter.clone();

    let count = Arc::new(Mutex::new(move |task: i32| {
        counter_clone.fetch_add(task as usize, Ordering::SeqCst);
    }));

    let pool = TaskPool::<i32>::new(2, count).await;
    pool.execute(5).await.expect("Task should be executed");
    pool.execute(3).await.expect("Task should be executed");

    // Give some time for tasks to be processed
    sleep(Duration::from_millis(100)).await;

    assert_eq!(counter.load(Ordering::SeqCst), 8);

    pool.shutdown().await;
}
